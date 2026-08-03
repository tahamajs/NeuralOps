from setuptools import find_packages, setup

setup(
    name="neuralops-agent",
    version="1.0.0",
    description="NeuralOps GPU Provider Agent – share your GPU and earn money",
    packages=find_packages(),
    install_requires=[
        "docker>=6.1.0",
        "websocket-client>=1.5.0",
        "psutil>=5.9.0",
        "speedtest-cli>=2.1.3",
        "requests>=2.31.0",
        "packaging>=23.0",
        "pyyaml>=6.0",
        "pynvml>=11.5.0",
    ],
    extras_require={
        "benchmark": ["torch>=2.1.0"],
    },
    entry_points={
        "console_scripts": [
            "neuralops-agent = agent.main:main",
        ],
    },
    python_requires=">=3.10",
)
