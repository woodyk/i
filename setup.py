#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# File: setup.py
# Author: Wadih Khairallah
# Description: 
# Created: 2024-12-02 15:18:59

from setuptools import setup, find_packages

# Read requirements from requirements.txt
def parse_requirements(filename):
    """Load requirements from a pip requirements file."""
    with open(filename, "r") as req_file:
        return [line.strip() for line in req_file if line.strip() and not line.startswith("#")]

setup(
    name="i",  # Name of your package
    version="0.1.0",
    packages=find_packages(),  # Automatically discover all packages
    install_requires=parse_requirements("requirements.txt"),  # Dependencies from requirements.txt
    description="A modular file, text and analysis cli tool.",
    author="Wadih Khariallah",
    author_email="woodyk@gmail.com",
    url="https://github.com/woodyk/i",
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    python_requires=">=3.6",  # Minimum Python version
)

