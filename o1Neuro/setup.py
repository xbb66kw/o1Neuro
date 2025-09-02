from setuptools import setup, find_packages

setup(
    name="o1Neuro",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "numba",
        "scikit-learn",
        "hyperopt",
    ],
)
