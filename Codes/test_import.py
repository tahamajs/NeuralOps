import sys, os
parent_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, parent_dir)
print("Importing framework.training.methods...")
from framework.training.methods import COMPRESSOR_REGISTRY
print("Done!")
