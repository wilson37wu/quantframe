from setuptools import setup, find_packages

setup(
    name="quantframe",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "matplotlib",
        "seaborn",
        "pyyaml",
        "pydantic",
        "pytest",
        "pytest-cov"
    ],
    python_requires=">=3.8",
    author="Wilson",
    description="A quantitative trading framework",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
