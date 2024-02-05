# copied from https://github.com/facebookresearch/CutLER/blob/main/cutler/modeling/meta_arch/__init__.py

from .build import META_ARCH_REGISTRY, build_model  # isort:skip

__all__ = list(globals().keys())
