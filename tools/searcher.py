"""
AgentClaw RAG 知识库检索模块
实现基于向量相似度的知识检索，支持文档加载、分块、嵌入和查询

架构:
    DocumentLoader -> TextSplitter -> VectorStore -> Retriever -> RAGEngine

特性:
    - 支持 TXT / Markdown / JSON / CSV 文档加载
    - 智能分块 (按段落/固定长度/重叠窗口)
    - 向量存储 (内存版, 可扩展为 FAISS/Chroma)
    - 相似度检索 (余弦相似度)
    - 与 Agent 系统集成 (作为 Level 2 工具)
    - 支持自定义 Embedding 函数

依赖:
    numpy (标准科学计算库)
可选:
    openai (用于文本嵌入)
"""

import hashlib
import json
import logging
import math
import re
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

# numpy 延迟加载 (~150MB)，仅在首次向量运算时导入
class _LazyNumpy:
    def __init__(self):
        self._module = None
    def _load(self):
        if self._module is None:
            import numpy as _m
            self._module = _m
        return self._module
    def __getattr__(self, n):
        return getattr(self._load(), n)

np = _LazyNumpy()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("RAGEngine")


# ============================================================
# 文档加载器
# ============================================================

class DocumentLoader:
    """
    文档加载器
    支持格式:
    - .txt: 纯文本
    - .md: Markdown (去除格式标记)
    - .json: JSON 文档 (提取所有字符串值)
    - .csv: CSV (逐行读取)
    """

    @staticmethod
    def load(file_path: str) -> list[dict[str, Any]]:
        """
        加载文档
        Args:
            file_path: 文件路径
        Returns:
            [{"content": str, "metadata": {"source": ..., "line": ...}}]
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        ext = path.suffix.lower()
        loader_map = {
            ".txt": DocumentLoader._load_text,
            ".md": DocumentLoader._load_markdown,
            ".json": DocumentLoader._load_json,
            ".csv": DocumentLoader._load_csv,
        }

        loader = loader_map.get(ext)
        if not loader:
            raise ValueError(f"不支持的文件格式: {ext} (支持: {list(loader_map.keys())})")

        docs = loader(file_path)
        logger.info(f"加载文档: {file_path} ({len(docs)} 个文档块)")
        return docs

    @staticmethod
    def _load_text(file_path: str) -> list[dict]:
        """加载纯文本"""
        content = Path(file_path).read_text(encoding="utf-8")
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
        return [
            {"content": p, "metadata": {"source": file_path, "type": "text"}}
            for p in paragraphs
        ]

    @staticmethod
    def _load_markdown(file_path: str) -> list[dict]:
        """加载 Markdown (去除格式)"""
        content = Path(file_path).read_text(encoding="utf-8")
        content = re.sub(r'^#{1,6}\s+', '', content, flags=re.MULTILINE)
        content = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', content)
        content = re.sub(r'```[\w]*\n?', '', content)
        content = re.sub(r'\*{1,2}([^*]+)\*{1,2}', r'\1', content)
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
        return [
            {"content": p, "metadata": {"source": file_path, "type": "markdown"}}
            for p in paragraphs
        ]

    @staticmethod
    def _load_json(file_path: str) -> list[dict]:
        """加载 JSON (提取所有字符串值)"""
        content = Path(file_path).read_text(encoding="utf-8")
        data = json.loads(content)

        def extract_strings(obj, prefix=""):
            results = []
            if isinstance(obj, dict):
                for k, v in obj.items():
                    results.extend(extract_strings(v, f"{prefix}.{k}"))
            elif isinstance(obj, list):
                for i, v in enumerate(obj):
                    results.extend(extract_strings(v, f"{prefix}[{i}]"))
            elif isinstance(obj, str) and len(obj) > 10:
                results.append(obj)
            return results

        strings = extract_strings(data)
        return [
            {"content": s, "metadata": {"source": file_path, "type": "json"}}
            for s in strings
        ]

    @staticmethod
    def _load_csv(file_path: str) -> list[dict]:
        """加载 CSV"""
        content = Path(file_path).read_text(encoding="utf-8")
        lines = [line.strip() for line in content.split("\n") if line.strip()]
        return [
            {"content": line, "metadata": {"source": file_path, "type": "csv", "row": i}}
            for i, line in enumerate(lines)
        ]

    @staticmethod
    def load_text_direct(text: str, source: str = "direct") -> list[dict]:
        """直接加载文本字符串"""
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if not paragraphs and text.strip():
            paragraphs = [text.strip()]
        return [
            {"content": p, "metadata": {"source": source, "type": "direct"}}
            for p in paragraphs
        ]


# ============================================================
# 文本分块器
# ============================================================

class TextSplitter:
    """
    文本分块器
    支持模式:
    - fixed: 固定长度分块
    - paragraph: 按段落分块 (保留段落完整性)
    - sliding: 滑动窗口 (带重叠)
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50, mode: str = "fixed"):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.mode = mode

    def split(self, documents: list[dict]) -> list[dict]:
        """分块文档列表"""
        chunks = []
        for doc in documents:
            content = doc["content"]
            metadata = doc.get("metadata", {})

            if self.mode == "fixed":
                doc_chunks = self._split_fixed(content)
            elif self.mode == "paragraph":
                doc_chunks = self._split_paragraph(content)
            elif self.mode == "sliding":
                doc_chunks = self._split_sliding(content)
            else:
                doc_chunks = self._split_fixed(content)

            for i, chunk_text in enumerate(doc_chunks):
                chunks.append({
                    "content": chunk_text,
                    "metadata": {
                        **metadata,
                        "chunk_index": i,
                        "chunk_total": len(doc_chunks),
                    },
                    "chunk_id": hashlib.md5(
                        f"{content[:50]}_{i}".encode()
                    ).hexdigest()[:12],
                })

        logger.info(
            f"分块完成: {len(documents)} -> {len(chunks)} 个块 "
            f"(模式: {self.mode}, 大小: {self.chunk_size})"
        )
        return chunks

    def _split_fixed(self, text: str) -> list[str]:
        """固定长度分块"""
        if len(text) <= self.chunk_size:
            return [text]

        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            if end < len(text):
                for sep in ["。", "！", "？", ".", "!", "?", "\n"]:
                    last_sep = text.rfind(sep, start, end)
                    if last_sep > start:
                        end = last_sep + 1
                        break
            chunks.append(text[start:end].strip())
            start = end - self.chunk_overlap
            if start >= len(text):
                break
        return [c for c in chunks if c]

    def _split_paragraph(self, text: str) -> list[str]:
        """按段落分块"""
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if not paragraphs:
            return [text.strip()] if text.strip() else []
        chunks = []
        current = ""
        for p in paragraphs:
            if len(current) + len(p) > self.chunk_size and current:
                chunks.append(current.strip())
                current = p
            else:
                current = f"{current}\n\n{p}" if current else p
        if current.strip():
            chunks.append(current.strip())
        return chunks

    def _split_sliding(self, text: str) -> list[str]:
        """滑动窗口分块"""
        if len(text) <= self.chunk_size:
            return [text]
        step = max(self.chunk_size - self.chunk_overlap, 1)
        chunks = []
        for start in range(0, len(text), step):
            end = start + self.chunk_size
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= len(text):
                break
        return chunks


# ============================================================
# 简易向量存储 (内存版) — 修复 OOM 版
# ============================================================

class InMemoryVectorStore:
    """
    内存版向量存储 (内存优化)
    - 词汇表上限 MAX_VOCAB_SIZE，防止 OOM
    - numpy float32 向量，比 Python list 省 7x 内存
    - numpy 矩阵运算，余弦相似度毫秒级
    - 支持自定义 Embedding 函数
    """

    MAX_VOCAB_SIZE = 5000   # 词汇表硬上限
    MAX_CHUNKS = 5000       # 最大文档块数

    def __init__(self, embedding_fn: Callable | None = None):
        self.embedding_fn = embedding_fn or self._default_embedding
        self._vectors: list[np.ndarray] = []   # numpy float32 向量列表
        self._documents: list[dict] = []
        self._vocab: dict[str, int] = {}       # word -> index
        self._idf: dict[str, float] = {}       # word -> idf weight
        self._built_vocab = False
        self._vocab_size = 0

    def add_documents(self, chunks: list[dict]):
        """添加文档块到向量库"""
        if not chunks:
            return

        # 文档块数量保护
        if len(self._documents) + len(chunks) > self.MAX_CHUNKS:
            overflow = len(self._documents) + len(chunks) - self.MAX_CHUNKS
            chunks = chunks[:len(chunks) - overflow]
            logger.warning(f"文档块已达上限 {self.MAX_CHUNKS}，截断 {overflow} 个块")

        if self.embedding_fn == self._default_embedding:
            if not self._built_vocab:
                self._build_vocab([c["content"] for c in chunks])
            else:
                # 增量更新: 合并已有文档 + 新文档重建词表
                self._rebuild_vocab([c["content"] for c in chunks])

        for chunk in chunks:
            vec = self.embedding_fn(chunk["content"])
            self._vectors.append(vec)
            self._documents.append(chunk)

        mem_mb = self._estimate_memory()
        logger.info(f"向量库已更新: {len(self._documents)} 个文档块, "
                     f"词表: {self._vocab_size}, 内存: ~{mem_mb:.1f}MB")

    def search(self, query: str, top_k: int = 5) -> list[tuple[dict, float]]:
        """相似度检索, 按相似度降序返回 (numpy 向量化)"""
        if not self._vectors:
            return []

        query_vec = self.embedding_fn(query)
        if not isinstance(query_vec, np.ndarray):
            query_vec = np.array(query_vec, dtype=np.float32)

        # 一次性矩阵运算: 所有文档向量 vs 查询向量
        mat = np.array(self._vectors, dtype=np.float32)   # (N, V)
        dots = mat @ query_vec                              # (N,)
        norm_doc = np.linalg.norm(mat, axis=1)              # (N,)
        norm_q = np.linalg.norm(query_vec)

        # 避免除零
        with np.errstate(divide='ignore', invalid='ignore'):
            sims = np.where(
                (norm_doc > 0) & (norm_q > 0),
                dots / (norm_doc * norm_q),
                0.0
            )

        # 取 top_k
        actual_k = min(top_k, len(sims))
        top_idx = np.argsort(sims)[::-1][:actual_k]
        results = [(self._documents[i], float(sims[i])) for i in top_idx if sims[i] > 0]

        logger.info(f"检索完成: {len(self._vectors)} 个块中找到 {len(results)} 条结果")
        return results

    def _tokenize(self, text: str) -> list[str]:
        r"""分词：英文单词 + 中文字符 bigram

        英文/数字按 \w+ 分词，中文提取连续汉字段后生成字符 bigram。
        例如 "数字孪生系统" -> ["数字", "字孪", "孪生", "生系", "系统"]
        这样查询 "数字孪生" 时能匹配到文档中的 "数字孪生系统"（共享 bigram）。
        """
        words = re.findall(r'\w+', text.lower())

        # 提取中文连续字符段，生成 bigram
        chinese_segments = re.findall(r'[\u4e00-\u9fff]+', text)
        for segment in chinese_segments:
            if len(segment) >= 2:
                for i in range(len(segment) - 1):
                    bigram = segment[i:i+2]
                    if bigram not in words:
                        words.append(bigram)
            elif len(segment) == 1:
                words.append(segment)

        return words

    def _build_vocab(self, texts: list[str]):
        """构建 TF-IDF 词汇表 (带词频上限，含中文 bigram)"""
        doc_count = len(texts)
        word_df = Counter()
        for text in texts:
            words = set(self._tokenize(text))
            word_df.update(words)

        total_words = len(word_df)
        # 只保留最高频的 MAX_VOCAB_SIZE 个词
        most_common = word_df.most_common(self.MAX_VOCAB_SIZE)
        self._vocab = {word: idx for idx, (word, _) in enumerate(most_common)}
        self._vocab_size = len(self._vocab)

        # IDF 计算
        self._idf = {}
        for word, df in most_common:
            self._idf[word] = math.log(doc_count / (1 + df)) + 1

        self._built_vocab = True
        logger.info(f"词表构建完成: {self._vocab_size} 词 "
                     f"(原始 {total_words} 词, 截断 {total_words - self._vocab_size})")

    def _rebuild_vocab(self, new_texts: list[str]):
        """增量添加文档时重建词表 + 重新向量化已有文档"""
        all_texts = [d["content"] for d in self._documents] + new_texts
        old_count = len(self._documents)

        self._built_vocab = False
        self._build_vocab(all_texts)

        # 用新词表重新向量化所有已有文档
        self._vectors = [self.embedding_fn(d["content"]) for d in self._documents]
        logger.info(f"词表重建: 已重新向量化 {old_count} 个已有文档块")

    def _default_embedding(self, text: str) -> np.ndarray:
        """
        默认向量化: TF-IDF + 中文 bigram (numpy float32)
        注意: 这不是真正的深度学习 Embedding, 仅用于演示。
        生产环境建议使用:
        - OpenAI: text-embedding-3-small
        - 智谱: embedding-3
        - BGE: bge-large-zh
        """
        words = self._tokenize(text)
        tf = Counter(words)
        total = max(len(words), 1)

        if self._vocab_size == 0:
            return np.zeros(10, dtype=np.float32)

        vec = np.zeros(self._vocab_size, dtype=np.float32)
        for word, count in tf.items():
            if word in self._vocab:
                idx = self._vocab[word]
                vec[idx] = (count / total) * self._idf.get(word, 1.0)
        return vec

    def _estimate_memory(self) -> float:
        """估算当前向量库内存占用 (MB)"""
        if not self._vectors:
            return 0.0
        # 每个向量: vocab_size * 4 bytes (float32)
        vec_mem = len(self._vectors) * self._vocab_size * 4
        return vec_mem / (1024 * 1024)

    def _cosine_similarity(self, a, b) -> float:
        """余弦相似度 (兼容非 numpy 向量的 fallback)"""
        if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
            dot = np.dot(a, b)
            na = np.linalg.norm(a)
            nb = np.linalg.norm(b)
            return float(dot / (na * nb)) if na > 0 and nb > 0 else 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    @property
    def doc_count(self) -> int:
        return len(self._documents)

    def clear(self):
        self._vectors.clear()
        self._documents.clear()
        self._vocab.clear()
        self._idf.clear()
        self._built_vocab = False
        self._vocab_size = 0


# ============================================================
# RAG 检索引擎
# ============================================================

class RAGEngine:
    """
    RAG 检索引擎
    将文档加载、分块、向量存储、检索整合为一个完整的 RAG 管道。

    使用示例:
        rag = RAGEngine()
        rag.add_documents("knowledge_base/api_doc.md")
        rag.add_text("自定义知识文本...")
        results = rag.search("如何调用API?")
        for doc, score in results:
            print(f"[{score:.3f}] {doc['content'][:100]}")
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50,
                 embedding_fn: Callable | None = None):
        self.loader = DocumentLoader()
        self.splitter = TextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, mode="fixed")
        self.vector_store = InMemoryVectorStore(embedding_fn)
        self._sources: list[str] = []

    def add_documents(self, file_path: str) -> int:
        """加载文档并加入向量库"""
        docs = self.loader.load(file_path)
        chunks = self.splitter.split(docs)
        self.vector_store.add_documents(chunks)
        self._sources.append(file_path)
        return len(chunks)

    def add_text(self, text: str, source: str = "direct") -> int:
        """直接添加文本到知识库"""
        docs = self.loader.load_text_direct(text, source)
        chunks = self.splitter.split(docs)
        self.vector_store.add_documents(chunks)
        return len(chunks)

    def search(self, query: str, top_k: int = 5) -> list[tuple[dict, float]]:
        """检索知识库"""
        if self.vector_store.doc_count == 0:
            logger.warning("知识库为空，请先添加文档")
            return []
        results = self.vector_store.search(query, top_k)
        logger.info(f"检索: '{query[:30]}...' -> {len(results)} 条结果")
        return results

    def search_as_context(self, query: str, top_k: int = 3, max_chars: int = 2000) -> str:
        """检索并格式化为 LLM 上下文"""
        results = self.search(query, top_k)
        if not results:
            return "知识库中未找到相关信息。"
        context_parts = ["以下是从知识库中检索到的相关信息:\n"]
        total_chars = 0
        for i, (doc, score) in enumerate(results, 1):
            content = doc["content"]
            source = doc.get("metadata", {}).get("source", "未知")
            if total_chars + len(content) > max_chars:
                content = content[:max_chars - total_chars]
            context_parts.append(f"[{i}] (相关度: {score:.2f}, 来源: {source})")
            context_parts.append(content)
            context_parts.append("")
            total_chars += len(content)
            if total_chars >= max_chars:
                break
        return "\n".join(context_parts)

    @property
    def doc_count(self) -> int:
        return self.vector_store.doc_count

    @property
    def sources(self) -> list[str]:
        return self._sources.copy()

    def get_stats(self) -> dict[str, Any]:
        """获取统计信息"""
        return {
            "doc_count": self.doc_count,
            "sources": self._sources,
            "chunk_size": self.splitter.chunk_size,
            "chunk_overlap": self.splitter.chunk_overlap,
        }


# ============================================================
# 与 Agent 系统集成的工具接口
# ============================================================

def create_rag_tool(rag_engine: RAGEngine) -> dict[str, Any]:
    """
    创建可注册到 ToolRegistry 的 RAG 工具
    返回格式符合 AgentClaw Level 2 的 tool 注册接口。

    使用:
        from tools.registry import registry
        rag = RAGEngine()
        rag.add_documents("knowledge/base.md")
        tool_def = create_rag_tool(rag)
        registry.register_func(tool_def["func"], **tool_def["info"])
    """

    def knowledge_search(query: str, top_k: int = 3) -> dict:
        """搜索知识库获取相关信息"""
        try:
            results = rag_engine.search(query, top_k=top_k)
            if not results:
                return {"success": True, "result": "未找到相关信息", "matches": 0}
            formatted = []
            for i, (doc, score) in enumerate(results, 1):
                formatted.append({
                    "rank": i,
                    "score": round(score, 4),
                    "content": doc["content"][:300],
                    "source": doc.get("metadata", {}).get("source", "未知"),
                })
            return {
                "success": True,
                "result": f"找到 {len(results)} 条相关信息",
                "matches": len(results),
                "details": formatted,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    return {
        "func": knowledge_search,
        "info": {
            "name": "knowledge_search",
            "description": "搜索知识库获取相关信息。用于回答需要专业知识支撑的问题。",
            "parameters": [
                {"name": "query", "type": "string", "description": "搜索查询文本"},
                {"name": "top_k", "type": "number", "description": "返回结果数量, 默认3",
                 "required": False, "default": 3},
            ],
            "category": "search",
        },
    }


# ============================================================
# 快速演示
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("AgentClaw RAG 知识库检索演示")
    print("=" * 60)

    rag = RAGEngine(chunk_size=200)

    knowledge = """
    AgentClaw 是一个 Python AI Agent 框架, 分为4个等级。
    Level 1 基础问答: 使用 DeepSeek API 实现对话功能。
    Level 2 工具增强: 集成计算器、文件读写、命令执行等工具。
    Level 3 多Agent协作: 使用 AgentOrchestrator 实现多Agent调度。
    Level 4 自主进化: 通过反思循环自动优化 Agent 行为。
    ToolRegistry 是工具注册中心, 支持 @register 装饰器注册工具。
    安全机制包括路径白名单、文件黑名单、命令白名单和危险模式检测。
    所有安全检查必须 raise PermissionError, 不能 return error dict。
    多模态支持包括 VLM 视觉理解和图片生成, 使用 glm-4.6v-flash 和 CogView-3-Flash。
    RAG 模块支持 TXT/Markdown/JSON 文档加载和 TF-IDF 向量检索。
    """

    count = rag.add_text(knowledge, source="AgentClaw知识库")
    print(f"\n已添加 {count} 个知识块到向量库")

    queries = [
        "AgentClaw 有哪些等级?",
        "安全机制是怎么工作的?",
        "ToolRegistry 怎么用?",
        "什么是多模态?",
    ]

    for query in queries:
        print(f"\n{'='*40}")
        print(f"查询: {query}")
        results = rag.search(query, top_k=2)
        for doc, score in results:
            print(f"  [{score:.3f}] {doc['content'][:80]}...")

    print(f"\n{'='*40}")
    print("LLM 上下文格式:")
    context = rag.search_as_context("安全机制", top_k=2)
    print(context)
