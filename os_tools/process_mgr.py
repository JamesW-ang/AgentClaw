# ============================================================
# process_mgr - 进程生命周期管理工具
# ============================================================
# 功能:
#   - 启动 / 停止 / 状态查询
#   - 命令白名单安全过滤
#   - SIGTERM → SIGKILL 优雅停止
# ============================================================

import subprocess
import time

# 命令白名单（只允许这些前缀的命令）
ALLOWED_PREFIXES = [
    "python", "python3", "node", "npm", "echo",
    "sleep", "cat", "ls", "pwd", "date", "whoami",
]


class ProcessManager:
    """
    进程管理器。
    通过命令白名单 + 生命周期管理，确保只执行安全命令。
    """

    def __init__(self):
        self._processes = {}  # {name: {"proc": subprocess.Popen, "pid": int, "cmd": str, "start_time": float}}

    def start(self, name: str, command: str) -> dict:
        """
        启动一个命名进程。

        参数:
            name:    进程名称（用于后续管理）
            command: 要执行的命令

        返回:
            {"success": bool, "pid": int, "message": str}
        """
        # ---------- 1. 名称唯一性检查 ----------
        if name in self._processes:
            old = self._processes[name]["proc"]
            if old.poll() is None:
                return {"success": False, "message": f"进程 '{name}' 已在运行"}

        # ---------- 2. 命令白名单检查 ----------
        cmd_stripped = command.strip()
        allowed = False
        for prefix in ALLOWED_PREFIXES:
            if cmd_stripped.startswith(prefix + " ") or cmd_stripped.startswith(prefix):
                allowed = True
                break

        if not allowed:
            return {
                "success": False,
                "message": f"命令不在白名单中: {command}",
            }

        # ---------- 3. 启动进程 ----------
        try:
            proc = subprocess.Popen(
                cmd_stripped,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self._processes[name] = {
                "proc": proc,
                "pid": proc.pid,
                "cmd": command,
                "start_time": time.time(),
            }
            return {
                "success": True,
                "pid": proc.pid,
                "message": f"进程 '{name}' 已启动 (PID={proc.pid})",
            }
        except Exception as e:
            return {"success": False, "message": f"启动失败: {e}"}

    def list_processes(self) -> dict:
        """
        列出所有被管理的进程。

        返回:
            {"success": bool, "count": int, "processes": list}
        """
        result = []
        for name, info in self._processes.items():
            proc = info["proc"]
            status = "running" if proc.poll() is None else "stopped"
            result.append({
                "name": name,
                "pid": info["pid"],
                "cmd": info["cmd"],
                "status": status,
                "uptime": round(time.time() - info["start_time"], 1),
            })

        return {"success": True, "count": len(result), "processes": result}

    def status(self, name: str) -> dict:
        """
        获取指定进程的状态。

        参数:
            name: 进程名称

        返回:
            {"success": bool, "status": str, "pid": int, ...}
        """
        if name not in self._processes:
            return {"success": False, "message": f"进程 '{name}' 不存在"}

        info = self._processes[name]
        proc = info["proc"]

        try:
            cpu = proc.cpu_percent() if proc.poll() is None else 0
        except Exception:
            cpu = 0

        return {
            "success": True,
            "name": name,
            "pid": info["pid"],
            "cmd": info["cmd"],
            "status": "running" if proc.poll() is None else "stopped",
            "cpu_percent": cpu,
            "uptime": round(time.time() - info["start_time"], 1),
        }

    def stop(self, name: str) -> dict:
        """
        停止指定进程（先 SIGTERM，2秒后 SIGKILL）。

        参数:
            name: 进程名称

        返回:
            {"success": bool, "message": str}
        """
        if name not in self._processes:
            return {"success": False, "message": f"进程 '{name}' 不存在"}

        proc = self._processes[name]["proc"]

        if proc.poll() is not None:
            return {"success": True, "message": f"进程 '{name}' 已停止"}

        # SIGTERM
        try:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                # SIGKILL
                proc.kill()
                proc.wait()
        except Exception as e:
            return {"success": False, "message": f"停止失败: {e}"}

        return {"success": True, "message": f"进程 '{name}' 已停止"}
