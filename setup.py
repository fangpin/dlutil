from setuptools import setup, find_packages

setup(
    name="dlutil",
    version="0.1.0",
    license="MIT",
    author="Pin Fang",
    author_email="fpfangpin@hotmail.com",
    url="https://github.com/fangpin/dlutil",
    description="An collection of useful util function for deeplearning",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)
