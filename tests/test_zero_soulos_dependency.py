"""零 Soul OS 依赖 enforcement（不只文档声明，用测试证明）。"""

from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1] / "src" / "soul_elevation"

# 本阶段源码实际使用的标准库模块白名单。
STDLIB_USED = {"__future__", "abc", "dataclasses", "typing", "uuid"}


def _py_files():
    return sorted(SRC_ROOT.rglob("*.py"))


def test_package_imports_are_stdlib_only():
    """包内所有非相对 import 必须来自标准库 —— 即零 Soul OS / 零第三方依赖。"""
    for path in _py_files():
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            s = line.strip()
            if not (s.startswith("import ") or s.startswith("from ")):
                continue
            parts = s.split()
            if len(parts) < 2:
                continue
            top = parts[1].split(".")[0]
            if top == "":  # 相对导入（from .models import ...），包内合法
                continue
            assert top in STDLIB_USED, f"{path.name}:{lineno} 非标准库/Soul OS import: {s}"


def test_no_soul_os_module_roots():
    """显式禁止 Soul OS 模块根（inner_life / memory / sage / soul / world 等）。"""
    forbidden = ("soul_os", "inner_life", "memory", "sage", "world", "soul", "narrative", "trigger", "provenance")
    for path in _py_files():
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            s = line.strip()
            if not (s.startswith("import ") or s.startswith("from ")):
                continue
            parts = s.split()
            if len(parts) >= 2 and parts[1].split(".")[0] in forbidden:
                raise AssertionError(f"{path.name}:{lineno} 引用了 Soul OS 模块: {s}")
