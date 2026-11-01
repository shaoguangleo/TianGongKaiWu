"""旧模块兼容入口；实现统一位于 tiangong.model，使用前请安装项目。"""
from importlib import import_module
import sys

_implementation = import_module("tiangong.model")

# 同一模块对象，避免两套实现和旧调用者的 patch 失效。
sys.modules[__name__] = _implementation
