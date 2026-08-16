#!/usr/bin/env python3
"""
Setup and packaging configuration for NeuralOps GPU Provider Agent.
"""

from pathlib import Path
from setuptools import find_packages, setup

README = Path(__file__).resolve().parent / "README.md"
long_description = README.read_text(encoding="utf-8") if README.exists() else ""

setup(
    name="neuralops-agent",
    version="1.0.0",
    description="NeuralOps GPU Provider Agent — share your GPU compute and earn money",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="NeuralOps Team",
    author_email="support@neuralops.com",
    url="https://github.com/neuralops/neuralops-agent",
    license="MIT",
    packages=find_packages(include=["agent", "agent.*", "tests", "tests.*"]),
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.png", "*.ico", "*.icns"],
    },
    install_requires=[
        "docker>=6.1.0",
        "websocket-client>=1.5.0",
        "psutil>=5.9.0",
        "speedtest-cli>=2.1.3",
        "requests>=2.31.0",
        "packaging>=23.0",
        "pyyaml>=6.0",
        "pynvml>=11.5.0",
        "PySocks>=1.7.1",
    ],
    extras_require={
        "benchmark": ["torch>=2.1.0"],
        "dev": ["pytest>=7.0.0", "pytest-cov>=4.0.0", "black>=23.0", "flake8>=6.0"],
        "test": ["pytest>=7.0.0"],
    },
    entry_points={
        "console_scripts": [
            "neuralops-agent = agent.main:main",
            "neuralops-bridge = bridge:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: System :: Distributed Computing",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.14",
    ],
    python_requires=">=3.10",
)
