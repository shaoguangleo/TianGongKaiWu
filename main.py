"""旧模块兼容入口；实现统一位于 tiangong.main，使用前请安装项目。"""
from importlib import import_module
import sys

_implementation = import_module("tiangong.main")

if __name__ == "__main__":
    _implementation.main()
else:
    # 模块别名保留旧调用者的私有符号与 monkeypatch 行为。
    sys.modules[__name__] = _implementation
