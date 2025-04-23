from setuptools import setup, find_packages

setup(
    name="py_op_beta",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "streamlit",
        "yfinance"
    ],  # Add dependencies if needed
)