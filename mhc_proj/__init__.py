# When the package is built using Pytorch, we typically
# need to import torch first to properly load some libraries
# such as libc10.so
try:
    import torch
except:
    pass

from ._internal import *
from .module import MHCProjectionN4, MHCSinkhornN4

__version__ = "0.1.0"
