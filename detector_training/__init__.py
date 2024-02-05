# copied from https://github.com/facebookresearch/CutLER/blob/main/cutler/__init__.py

import config
import engine
import modeling
import structures
import tools
import demo 

# dataset loading
from . import data  # register all new datasets
from data import datasets  # register all new datasets
from solver import *

# from .data import register_all_imagenet