# ADR-008: ChromaDB + bge-small-zh for Semantic RAG

**状态**：已采纳 | **日期**：2026-05 | **决定者**：架构组

## 背景

RAG 系统需要支持中文文档的语义搜索。原有 TF-IDF 实现（`tools/searcher.py`）只能做关键词匹配，无法理解语义。README 宣称使用 ChromaDB + bge-small-zh，但实际未完整接入。

## 方案

部署 **ChromaDB + bge-small-zh-v1.5** 作为主 RAG 后端，保留 TF-IDF 作为降级：

```
文档 → 加载(loader) → 切分(splitter) → ChromaDB(embedding写入)
查询 → ChromaDB(语义检索) → [结果不足?] → TF-IDF(关键词兜底)
```

## 考虑过的选项

### 选项 A：仅 TF-IDF（现状）
- 优点：零依赖、极快（纯 Python 计数）
- 缺点：无法理解语义、同义词不匹配、"价格"搜不到"费用"

### 选项 B：ChromaDB + bge-large-zh
- 优点：更高精度（MTEB 63.7 vs 58.3）
- 缺点：1.3GB 模型 vs 33MB，40 倍资源消耗，收益仅 +5%

### 选项 C：ChromaDB + bge-small-zh（选定）
- 优点：精度足够 + 极快推理（CPU 2ms）+ 小内存

## Embedding 选型比较

| 模型 | 大小 | 维度 | CPU 推理 | MTEB 中文 |
|------|------|------|----------|-----------|
| bge-small-zh-v1.5 | 33MB | 512 | ~2ms | 58.3 |
| bge-base-zh-v1.5 | 212MB | 768 | ~10ms | 61.2 |
| bge-large-zh-v1.5 | 1.3GB | 1024 | ~20ms | 63.7 |
| text-embedding-3-small | API | 1536 | ~100ms | ~62 (英文优化) |

## 后果

- 正：中文语义搜索质量显著提升（"如何安装" 也能搜到 "部署指南"）
- 正：33MB 模型可嵌入 Docker 镜像，无需外部 API
- 正：TF-IDF 兜底确保 ChromaDB 不可用时服务不中断
- 负：首次加载需下载模型（~33MB）
- 负：内存增加 ~200MB（加载模型 + ChromaDB）

## 关联

- `tools/builtin.py`：`knowledge_search` 工具使用了双后端 RAG
- `tools/searcher.py`：TF-IDF 实现保留为 fallback
- `core/config.py`：`CHROMA_DB_PATH` 配置项
