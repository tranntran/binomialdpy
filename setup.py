from os import path

from setuptools import find_packages, setup

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="binomialdpy",
    packages=find_packages(where="binomialdpy"),
    version="0.1.3",
    license='GPLv3',
    description="Differentially Private UMP Test for Binomial Data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Jordan Awan, Aleksandra Slavkovic, Tran Tran, and Kaitlyn Dowden",
    author_email="ttran2401@gmail.com",
    url="https://github.com/tranntran/binomialdpy",
    keywords=["Differential Privacy", "Binomial Data", "Inference"],
    install_requires=["scipy>=1.7.0", "numpy", "pandas"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Intended Audience :: Science/Research",
    ],
)