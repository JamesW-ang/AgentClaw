# ============================================================
# scheduler - 定时任务调度工具
# ============================================================
# 功能:
#   - 间隔任务（每隔 N 秒执行）
#   - Cron 任务（标准 5 位 Cron 表达式）
#   - 暂停 / 恢复 / 删除任务
#   - 优雅关闭
# ============================================================

import threading
import time
import uuid


class TaskScheduler:
    """
    轻量级任务调度器（基于 threading，无需外部依赖）。

    支持:
    - 间隔任务: add_interval_task(name, func, seconds=1)
    - Cron 任务:  add_cron_task(name, func, cron_expr="* * * * *")
    """

    def __init__(self):
        self._jobs = {}        # {job_id: {"name": str, "func": callable, "interval": float, "paused": bool, "thread": Thread}}
        self._lock = threading.Lock()
        self._running = True

    @staticmethod
    def _parse_cron(cron_expr: str) -> dict:
        """
        解析标准 5 位 Cron 表达式: 分 时 日 月 周

        返回:
            {"min": set, "hour": set, "day": set, "month": set, "dow": set}
        """
        parts = cron_expr.strip().split()
        if len(parts) != 5:
            raise ValueError(f"Cron 表达式需要 5 段，得到 {len(parts)} 段: {cron_expr}")

        def parse_field(field, min_val, max_val):
            if field == "*":
                return set(range(min_val, max_val + 1))
            result = set()
            for item in field.split(","):
                if "-" in item:
                    start, end = item.split("-")
                    result.update(range(int(start), int(end) + 1))
                else:
                    result.add(int(item))
            return result

        return {
            "min": parse_field(parts[0], 0, 59),
            "hour": parse_field(parts[1], 0, 23),
            "day": parse_field(parts[2], 1, 31),
            "month": parse_field(parts[3], 1, 12),
            "dow": parse_field(parts[4], 0, 6),
        }

    def _run_loop(self, job_id: str):
        """任务执行循环。"""
        job = self._jobs.get(job_id)
        if not job:
            return

        if job.get("cron_parsed"):
            # Cron 模式: 每秒检查一次是否匹配
            while self._running and not job["stopped"].is_set():
                if not job["paused"]:
                    now = time.localtime()
                    cron = job["cron_parsed"]
                    if (now.tm_min in cron["min"] and
                        now.tm_hour in cron["hour"] and
                        now.tm_mday in cron["day"] and
                        now.tm_mon in cron["month"] and
                        now.tm_wday in cron["dow"]):
                        try:
                            job["func"]()
                        except Exception:
                            pass
                time.sleep(1)
        else:
            # 间隔模式
            interval = job["interval"]
            while self._running and not job["stopped"].is_set():
                if not job["paused"]:
                    try:
                        job["func"]()
                    except Exception:
                        pass
                # 每 0.1 秒检查一次是否停止，避免长时间阻塞
                for _ in range(int(interval * 10)):
                    if not self._running or job["stopped"].is_set():
                        break
                    time.sleep(0.1)

    def add_interval_task(self, name: str, func: callable, seconds: int = 1) -> dict:
        """
        添加间隔任务。

        参数:
            name:    任务名称
            func:    回调函数
            seconds: 执行间隔（秒）

        返回:
            {"success": bool, "job_id": str}
        """
        job_id = str(uuid.uuid4())[:8]
        job = {
            "name": name,
            "func": func,
            "interval": seconds,
            "paused": False,
            "stopped": threading.Event(),
            "thread": None,
        }
        with self._lock:
            self._jobs[job_id] = job

        t = threading.Thread(target=self._run_loop, args=(job_id,), daemon=True)
        job["thread"] = t
        t.start()

        return {"success": True, "job_id": job_id, "message": f"间隔任务 '{name}' 已添加"}

    def add_cron_task(self, name: str, func: callable, cron_expr: str = "* * * * *") -> dict:
        """
        添加 Cron 任务。

        参数:
            name:      任务名称
            func:      回调函数
            cron_expr: 标准 5 位 Cron 表达式（分 时 日 月 周）

        返回:
            {"success": bool, "job_id": str}
        """
        try:
            cron_parsed = self._parse_cron(cron_expr)
        except ValueError as e:
            return {"success": False, "message": str(e)}

        job_id = str(uuid.uuid4())[:8]
        job = {
            "name": name,
            "func": func,
            "paused": False,
            "stopped": threading.Event(),
            "cron_parsed": cron_parsed,
            "thread": None,
        }
        with self._lock:
            self._jobs[job_id] = job

        t = threading.Thread(target=self._run_loop, args=(job_id,), daemon=True)
        job["thread"] = t
        t.start()

        return {"success": True, "job_id": job_id, "message": f"Cron 任务 '{name}' 已添加"}

    def list_tasks(self) -> dict:
        """
        列出所有任务。

        返回:
            {"success": bool, "count": int, "tasks": list}
        """
        tasks = []
        with self._lock:
            for job_id, job in self._jobs.items():
                tasks.append({
                    "job_id": job_id,
                    "name": job["name"],
                    "paused": job["paused"],
                    "alive": job["thread"].is_alive() if job["thread"] else False,
                })

        return {"success": True, "count": len(tasks), "tasks": tasks}

    def pause_task(self, job_id: str) -> dict:
        """
        暂停指定任务。

        参数:
            job_id: 任务 ID

        返回:
            {"success": bool, "message": str}
        """
        with self._lock:
            job = self._jobs.get(job_id)
        if not job:
            return {"success": False, "message": f"任务 {job_id} 不存在"}

        job["paused"] = not job["paused"]
        state = "已暂停" if job["paused"] else "已恢复"
        return {"success": True, "message": f"任务 '{job['name']}' {state}"}

    def shutdown(self):
        """优雅关闭调度器，停止所有任务。"""
        self._running = False
        with self._lock:
            for job in self._jobs.values():
                job["stopped"].set()
        # 等待线程结束
        for job in self._jobs.values():
            if job["thread"]:
                job["thread"].join(timeout=3)
        self._jobs.clear()
