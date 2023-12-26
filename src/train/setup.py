# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from setuptools import find_packages, setup

setup(
    name="pytrain",
    version="0.0.1",
    description="Framework for training models on private data",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    packages=find_packages(),
    entry_points={
        "console_scripts": ["pytrain=pytrain.pipeline_executor:main"],
    },
    python_requires=">=3.8",
    install_requires=[
    ],
)
