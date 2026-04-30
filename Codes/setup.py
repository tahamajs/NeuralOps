from setuptools import setup, find_packages

setup(
    name="svd-lin-studio",
    version="2.0.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.36.0",
        "datasets>=2.14.0",
        "evaluate>=0.4.0",
        "numpy>=1.24.0",
        "pandas>=1.5.0",
        "scikit-learn>=1.3.0",
        "plotly>=5.10.0",
        "pyyaml>=6.0",
    ],
    entry_points={
        "console_scripts": [
            "svd-lin=studio.cli:main",
        ]
    },
)