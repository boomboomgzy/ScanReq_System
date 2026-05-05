import os
import subprocess
import shutil
import platform


def _resolve_apktool_command():
    """在不同环境中解析 apktool 的可执行命令。
    优先顺序：
    1) 系统 PATH 中的 `apktool`
    2) 环境变量：`APKTOOL`, `APKTOOL_PATH`, `APKTOOL_HOME`
       - 指向具体文件：`.bat`/`.exe`/`.jar`
       - 指向目录：尝试寻找其中的 `apktool.bat` 或 `apktool.jar`
    返回：list 形式的命令数组，或 None 表示未找到。
    """
    # 1) which
    exe = shutil.which("apktool")
    if exe:
        return [exe]

    # 2) env vars
    candidates = [
        os.getenv("APKTOOL"),
        os.getenv("APKTOOL_PATH"),
        os.getenv("APKTOOL_HOME"),
    ]
    for c in candidates:
        if not c:
            continue
        c = c.strip('"')
        if os.path.isfile(c):
            # 文件路径
            if c.lower().endswith('.jar'):
                return ['java', '-jar', c]
            else:
                return [c]
        if os.path.isdir(c):
            # 目录路径 → 尝试查找常见文件
            bat = os.path.join(c, 'apktool.bat')
            jar = os.path.join(c, 'apktool.jar')
            exe2 = os.path.join(c, 'apktool.exe')
            if os.path.isfile(bat):
                return [bat]
            if os.path.isfile(exe2):
                return [exe2]
            if os.path.isfile(jar):
                return ['java', '-jar', jar]
    return None


def apktool_apk(apk_path: str, output_dir: str):
    """使用 apktool 反编译 APK 到指定输出目录。
    - 优先使用系统 PATH 中的 `apktool`；
    - 若未找到，支持通过环境变量 `APKTOOL`/`APKTOOL_PATH`/`APKTOOL_HOME` 指定路径；
    - 同时兼容 `.bat/.exe/.jar` 形式。
    """
    if not os.path.isfile(apk_path):
        print(f"❌ APK 文件不存在: {apk_path}")
        return False

    parent = os.path.dirname(output_dir)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)

    cmd_head = _resolve_apktool_command()
    if not cmd_head:
        print("❌ 未找到 'apktool'，请确保：\n - 已安装并在 PATH 中，或\n - 设置环境变量 APKTOOL / APKTOOL_PATH / APKTOOL_HOME 指向 apktool(.bat/.jar)。")
        return False

    try:
        result = subprocess.run(
            cmd_head + ["d", apk_path, "-o", output_dir, "-f"],
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
        )
        if result.returncode != 0:
            print("❌ apktool 执行失败:\n" + result.stderr)
            return False
        print("-- apktool 解析完成 --", flush=True)
        return True
    except FileNotFoundError:
        print("❌ 未找到 'apktool' 或 'java' 可执行文件，请检查 PATH 或环境变量配置。")
        return False

