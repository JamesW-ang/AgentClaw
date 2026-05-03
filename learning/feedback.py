# ============================================================
# 反馈信号收集系统
# ============================================================
# 功能: 收集和管理工具执行反馈信息
# 特性: 包含反馈数据结构和持久化存储
#      支持实时收集、查询和数据分析
# ============================================================

import json
import os
import time
from collections import deque
from dataclasses import asdict, dataclass, field

# ============================================================
# 反馈信号数据类
# ============================================================


@dataclass
class FeedbackSignal:
    """
    反馈信号 — 记录单次工具调用的执行反馈信息。

    该类捕获工具执行的关键指标，用于监控、优化和调试。

    属性:
        task_id (str): 任务唯一标识符
        tool_name (str): 调用的工具名称
        success (bool): 执行是否成功
        latency (float): 执行耗时（秒）
        error_type (str, 可选): 错误类型（如果发生错误）
        user_rating (float, 可选): 用户评分 (1-5 分)
        timestamp (float): 信号记录时间戳，默认为当前时间
        context (str): 执行上下文信息或备注
    """
    # 必需字段
    task_id: str                              # 任务唯一标识
    tool_name: str                            # 工具名称
    success: bool                             # 执行成功标志
    latency: float                            # 执行延迟（秒）

    # 可选字段
    error_type: str | None = None          # 错误类型 (如果有错误)
    user_rating: float | None = None       # 用户评分 (1-5)
    timestamp: float = field(default_factory=time.time)  # 时间戳 (自动设置为当前时间)
    context: str = ""                         # 执行上下文信息


# ============================================================
# 反馈收集器类
# ============================================================


class FeedbackCollector:
    """
    反馈收集器 — 管理反馈信号的收集、存储和查询。

    核心功能:
    1. 实时收集工具执行反馈信号
    2. 将反馈信号保存到 JSONL 文件以支持持久化
    3. 提供反馈查询接口 (按时间范围、工具类型等)
    4. 自动定期保存以防数据丢失

    特点:
    - 使用 deque 作为循环缓冲区，限制内存占用
    - 每 50 条信号自动保存一次，降低持久化开销
    - 支持 JSONL 格式，便于流式处理和分析

    属性:
        signals (deque): 循环缓冲队列，最多保存 10000 条信号
        persist_dir (str): 数据持久化目录路径
    """

    def __init__(self, persist_dir: str = "./evolution_data"):
        """
        初始化反馈收集器。

        参数:
            persist_dir (str, 可选): 数据保存目录，默认为 "./evolution_data"
                                    如果目录不存在会自动创建
        """
        # 创建循环缓冲队列 (最多保存 10000 条记录)
        self.signals: deque = deque(maxlen=10000)

        # 设置持久化目录
        self.persist_dir = persist_dir

        # 创建目录 (如果不存在)
        os.makedirs(persist_dir, exist_ok=True)

        # 从磁盘加载已保存的反馈数据
        self._load()

    def collect(self, signal: FeedbackSignal):
        """
        收集单条反馈信号。

        工作流程:
        1. 将信号添加到缓冲队列
        2. 每累积 50 条信号时自动保存到磁盘

        参数:
            signal (FeedbackSignal): 要收集的反馈信号
        """
        # 将信号添加到缓冲队列
        self.signals.append(signal)

        # 每 50 条信号自动持久化一次
        # 这样既能降低 I/O 开销，又能防止数据丢失
        if len(self.signals) % 50 == 0:
            self._save()

    def get_recent(self, n: int = 100) -> list:
        """
        获取最近的 N 条反馈信号。

        参数:
            n (int, 可选): 要获取的信号数量，默认为 100

        返回:
            list: 最近 N 条反馈信号的列表（按时间顺序从旧到新）
        """
        # 返回缓冲队列中最后 n 条信号
        return list(self.signals)[-n:]

    def get_by_tool(self, tool_name: str) -> list:
        """
        获取特定工具的所有反馈信号。

        用途: 分析某个工具的性能指标、成功率等。

        参数:
            tool_name (str): 工具名称

        返回:
            list: 该工具的所有反馈信号列表
        """
        # 过滤出与指定工具名称匹配的信号
        return [s for s in self.signals if s.tool_name == tool_name]

    def _save(self):
        """
        将缓冲区中的反馈信号保存到 JSONL 文件。

        工作流程:
        1. 转换每条反馈信号为字典格式
        2. 以 JSON 格式写入文件 (每行一条记录)
        3. 清空缓冲区以释放内存

        文件格式: JSONL (JSON Lines)
        - 每行是一个完整的 JSON 对象
        - 支持流式处理和逐行读取
        """
        # 构造反馈文件路径
        path = os.path.join(self.persist_dir, "feedback.jsonl")

        # 以追加模式打开文件
        with open(path, "a") as f:
            # 遍历缓冲区中的所有信号
            for s in self.signals:
                # 将反馈信号转换为字典
                signal_dict = asdict(s)
                # 写入 JSON 格式 (ensure_ascii=False 保持中文)
                f.write(json.dumps(signal_dict, ensure_ascii=False) + "\n")

        # 清空缓冲区
        self.signals.clear()

    def _load(self):
        """
        从 JSONL 文件中加载已保存的反馈信号。

        工作流程:
        1. 检查反馈文件是否存在
        2. 逐行读取 JSON 数据
        3. 重构反馈信号对象并加入缓冲区
        4. 自动处理异常行 (格式错误等)

        容错: 遇到无效的 JSON 行会跳过，不中断加载过程
        """
        # 构造反馈文件路径
        path = os.path.join(self.persist_dir, "feedback.jsonl")

        # 检查文件是否存在
        if os.path.exists(path):
            with open(path) as f:
                # 逐行读取文件
                for line in f:
                    try:
                        # 解析 JSON 行
                        data = json.loads(line.strip())
                        # 重构反馈信号对象
                        signal = FeedbackSignal(**data)
                        # 添加到缓冲区
                        self.signals.append(signal)
                    except Exception:
                        # 跳过无效行 (格式错误等)
                        pass
