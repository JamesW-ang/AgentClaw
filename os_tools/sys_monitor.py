# ============================================================
# sys_monitor - 系统资源监控工具
# ============================================================
# 功能:
#   - CPU / 内存 / 磁盘 / 网络实时监控
#   - 进程列表（按 CPU / 内存排序）
#   - macOS 兼容（无权限时优雅降级）
# ============================================================

import platform

# psutil 延迟加载 (~5MB)
_psutil = None

def _get_psutil():
    global _psutil
    if _psutil is None:
        import psutil as _psutil
    return _psutil


class SystemMonitor:
    """系统资源监控器，提供 CPU、内存、磁盘、网络的统一查询接口。"""

    def get_overview(self) -> dict:
        """
        获取系统概览信息。

        返回:
            {"success": bool, "cpu_percent": float, "memory": dict, ...}
        """
        try:
            cpu = _get_psutil().cpu_percent(interval=1)
            mem = _get_psutil().virtual_memory()

            return {
                "success": True,
                "cpu_percent": cpu,
                "cpu_count_logical": _get_psutil().cpu_count(logical=True),
                "cpu_count_physical": _get_psutil().cpu_count(logical=False),
                "memory": {
                    "total": mem.total,
                    "used": mem.used,
                    "available": mem.available,
                    "percent": mem.percent,
                },
                "platform": platform.system(),
                "hostname": platform.node(),
            }
        except Exception as e:
            return {"success": False, "message": f"获取系统概览失败: {e}"}

    def get_process_list(self, top_n: int = 10) -> dict:
        """
        获取占用资源最多的进程列表。

        参数:
            top_n: 返回前 N 个进程

        返回:
            {"success": bool, "total": int, "processes": list}
        """
        try:
            procs = []
            for p in _get_psutil().process_iter(["pid", "name", "cpu_percent", "memory_percent"]):
                try:
                    info = p.info
                    procs.append({
                        "pid": info["pid"],
                        "name": info["name"],
                        "cpu_percent": info["cpu_percent"] or 0,
                        "memory_percent": info["memory_percent"] or 0,
                    })
                except (_get_psutil().NoSuchProcess, _get_psutil().AccessDenied):
                    continue

            # 按 CPU 降序排序
            procs.sort(key=lambda x: x["cpu_percent"], reverse=True)

            return {
                "success": True,
                "total": len(procs),
                "processes": procs[:top_n],
            }
        except Exception as e:
            return {"success": False, "message": f"获取进程列表失败: {e}"}

    def get_disk_info(self) -> dict:
        """
        获取磁盘使用情况。

        返回:
            {"success": bool, "total": int, "used": int, "free": int, "percent": float}
        """
        try:
            disk = _get_psutil().disk_usage("/")
            return {
                "success": True,
                "total": disk.total,
                "used": disk.used,
                "free": disk.free,
                "percent": disk.percent,
            }
        except Exception as e:
            return {"success": False, "message": f"获取磁盘信息失败: {e}"}

    def get_network_stats(self) -> dict:
        """
        获取网络 IO 统计。

        返回:
            {"success": bool, "connections": int, "io": dict}
        """
        try:
            # macOS 上 net_connections() 需要管理员权限
            try:
                conn_count = len(_get_psutil().net_connections())
            except (_get_psutil().AccessDenied, PermissionError):
                conn_count = -1  # -1 表示无权限

            io = _get_psutil().net_io_counters()
            return {
                "success": True,
                "connections": conn_count,
                "io": {
                    "bytes_sent": io.bytes_sent,
                    "bytes_recv": io.bytes_recv,
                    "packets_sent": io.packets_sent,
                    "packets_recv": io.packets_recv,
                },
            }
        except Exception as e:
            return {"success": False, "message": f"获取网络统计失败: {e}"}
