from setuptools import setup, find_packages

setup(
    name="stg",
    version="0.3.0",
    description="Spatio-Temporal Graph infrastructure for event prediction markets (Polars-native)",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "polars>=0.20",
        "numpy>=1.22",
        "networkx>=3.0",
        "scikit-learn>=1.0",
    ],
    extras_require={
        "viz": ["matplotlib>=3.5"],
    },
)
