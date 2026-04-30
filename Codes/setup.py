# setup.py
from setuptools import setup, find_packages

setup(
    name="svd-lin",
    version="2.0.0",
    description="SVD-Lin: Linearizing Transformer Attention via Truncated SVD",
    author="SVD-Lin Team",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "datasets>=2.14.0",
        "evaluate>=0.4.0",
        "tokenizers>=0.13.0",
        "tqdm>=4.65.0",
        "numpy>=1.24.0",
        "pandas>=1.5.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "scikit-learn>=1.3.0",
        "scipy>=1.10.0",
        "optuna>=3.0.0",
        "plotly>=5.10.0",
        "pyyaml>=6.0",
    ],
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)