"""`janus` 命令行入口：一个总命令，下面挂子命令。

每个子命令是一个模块里的 ``main(argv) -> int``，按需导入：只跑 ``janus run`` 时不导入画图、分析的依赖。
加子命令只改下面的 ``COMMANDS``，并在 docs/reference/cli.md 补一行。
"""

from __future__ import annotations

import importlib
import sys

from janus import __version__

# 子命令 → (模块, 一句话说明)。模块要有 main(argv: list[str]) -> int。
COMMANDS: dict[str, tuple[str, str]] = {
    "run": ("janus.run", "读一份检查项目录，跑完全部检查，写 results.json 与报告"),
    "validate": ("janus.validate", "只校验检查项目录：展开后有哪些检查、载入时的校验规则过不过（不读运行目录）"),
    "demo": ("janus.demo", "造一组合成运行目录与检查项目录，不要 GPU / TPU，演示怎么判"),
    "curves": ("janus.analysis.curves", "多次运行按步对标的时序图：被测对 golden，或按设置分组互比"),
    "offsets": ("janus.analysis.offsets", "被测对 golden 的 loss 偏移归因：偏还是抖、哪个逐步量先漂"),
    "fault-matrix": ("janus.analysis.fault_matrix", "故障演练矩阵：哪条检查对哪种故障报警"),
    "drift": ("janus.analysis.drift", "多步漂移拆解：几组运行两两比、趋势包络、分组参数范数、输出层逐行"),
}


def _usage() -> str:
    width = max(len(n) for n in COMMANDS)
    lines = [
        "用法：janus <子命令> [参数…]    （等价写法：python -m janus <子命令> …）",
        "",
        "子命令：",
        *(f"  {n:<{width}}  {doc}" for n, (_, doc) in COMMANDS.items()),
        "",
        "每个子命令加 --help 看参数。janus --version 打印版本。",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(_usage(), file=sys.stdout if argv else sys.stderr)  # 要了帮助才算成功；什么都没给是用法错
        return 0 if argv else 2
    if argv[0] in ("-V", "--version", "version"):
        print(f"janus {__version__}")
        return 0
    name = argv[0]
    if name not in COMMANDS:
        if name.endswith((".yaml", ".yml")):  # 旧写法：python -m janus <目录文件> …，当 run 处理
            name, argv = "run", ["run", *argv]
        else:
            print(f"janus：没有子命令 {name!r}\n\n{_usage()}", file=sys.stderr)
            return 2
    module = importlib.import_module(COMMANDS[name][0])
    try:
        return int(module.main(argv[1:]) or 0)
    except OSError as e:  # 文件不存在、读不了：按约定退出码 2，不抛栈
        print(f"janus {name}：读不了文件：{e}", file=sys.stderr)
        return 2
    except Exception:  # noqa: BLE001  其他意外：打印栈方便排查，但退出码 2（1 只留给「有硬检查 fail」）
        import traceback

        traceback.print_exc()
        print(f"janus {name}：内部错误（见上面的栈）。退出码 2", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
