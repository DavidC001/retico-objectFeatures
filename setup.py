#!/usr/bin/env python

"""
Setup script for retico-objectFeatures
"""

import os
from setuptools import setup, find_packages

# Read version from version.py
version = {}
with open(os.path.join("retico_objectFeatures", "version.py")) as fp:
    exec(fp.read(), version)

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="retico-objectFeatures",
    version=version["__version__"],
    author="DavidC001",
    author_email="",
    description="A ReTiCo module for extracting visual features from detected objects using CLIP and other vision models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "retico-core",
        # "retico-vision",
        "torch>=1.9.0",
        "transformers>=4.20.0",
    ],
    keywords="retico, real-time, dialogue, object detection, computer vision, CLIP, feature extraction",
    project_urls={
        
    },
)