# ============================================================
# file_write - 安全文件写入工具
# ============================================================
# 功能:
#   - 基本文件写入和追加
#   - 覆盖时自动创建备份
#   - 路径黑名单（阻止写入敏感系统路径）
#   - 文件大小限制
#   - 危险内容检测（拦截 shell 炸弹等）
# ============================================================

import os
import shutil
import hashlib


# 安全配置
PATH_BLACKLIST = [
    "/etc/passwd", "/etc/shadow", "/etc/sudoers",
    "/etc/hosts", "/etc/crontab", "/etc/fstab",
    "/System", "/Library", "/usr/bin", "/usr/sbin",
    "/boot", "/sbin", "/bin",
]

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

DANGEROUS_PATTERNS = [
    "rm -rf /", "rm -rf /*", "mkfs", "dd if=/dev/zero",
    ":(){ :|:& };:", "fork bomb", "> /dev/sda",
    "chmod -R 777 /", "chown -R",
]


def file_write(path: str, content: str, mode: str = "write") -> dict:
    """
    安全文件写入。

    参数:
        path:   目标文件路径
        content: 要写入的内容
        mode:   "write"(覆盖) 或 "append"(追加)

    返回:
        {"success": bool, "size": int, "backup_created": bool, "message": str}
    """
    # ---------- 1. 路径黑名单检查 ----------
    abs_path = os.path.abspath(path)
    for blocked in PATH_BLACKLIST:
        if abs_path.startswith(blocked) or abs_path == blocked:
            return {
                "success": False,
                "message": f"安全拦截: 路径在黑名单中 ({blocked})",
            }

    # ---------- 2. 大小限制检查 ----------
    content_size = len(content.encode("utf-8"))
    if mode == "write" and content_size > MAX_FILE_SIZE:
        return {
            "success": False,
            "message": f"内容过大: {content_size} 字节 > {MAX_FILE_SIZE} 字节限制",
        }

    # ---------- 3. 危险内容检测 ----------
    content_lower = content.lower()
    for pattern in DANGEROUS_PATTERNS:
        if pattern.lower() in content_lower:
            return {
                "success": False,
                "message": f"安全拦截: 检测到危险内容 ({pattern})",
            }

    # ---------- 4. 自动备份（仅覆盖模式） ----------
    backup_created = False
    if mode == "write" and os.path.exists(abs_path):
        backup_path = abs_path + ".bak"
        shutil.copy2(abs_path, backup_path)
        backup_created = True

    # ---------- 5. 执行写入 ----------
    try:
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        write_mode = "a" if mode == "append" else "w"
        with open(abs_path, "w", encoding="utf-8") as f:
            if mode == "append":
                f.write(content)
            else:
                f.write(content)

        final_size = os.path.getsize(abs_path)
        return {
            "success": True,
            "size": final_size,
            "path": abs_path,
            "backup_created": backup_created,
            "message": "写入成功",
        }

    except Exception as e:
        return {
            "success": False,
            "message": f"写入失败: {e}",
        }
