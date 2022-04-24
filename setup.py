from setuptools import find_packages, setup

setup(
    name="jes",
    description="Joint entropy search",
    packages=find_packages(),
    install_requires=[
        "botorch",
        "pymoo",
    ],
    python_requires=">=3.7",
)