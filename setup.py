from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="alpharl-quant",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Production-grade ML pipeline for quantitative trading with reinforcement learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/alpharl-quant",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "yfinance>=0.2.0",
        "ta>=0.11.0",
        "scikit-learn>=1.3.0",
        "stable-baselines3>=2.0.0",
        "gymnasium>=0.29.0",
        "psycopg2-binary>=2.9.0",
        "sqlalchemy>=2.0.0",
        "pyyaml>=6.0",
        "python-json-logger>=2.0.0",
        "tqdm>=4.65.0",
        "python-dotenv>=1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "pytest-mock>=3.11.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.4.0",
        ],
        "viz": [
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
            "plotly>=5.14.0",
        ],
    },
)
