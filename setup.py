#!/usr/bin/env python3
"""Setup script."""
import os

from setuptools import setup


def open_file(fname):
    """Open and return a file-like object for the relative filename."""
    return open(os.path.join(os.path.dirname(__file__), fname))


setup(
    name="azul-smart-string-filter",
    description="using ai model to filter for useful strings",
    author="Azul",
    author_email="azul@asd.gov.au",
    url="https://www.asd.gov.au/",
    packages=["azul_smart_string_filter"],
    package_data={
        "azul_smart_string_filter": ["model/model.onnx", "model/vectorizer.json"],
    },
    include_package_data=True,
    python_requires=">=3.12",
    classifiers=[],
    entry_points={
        "console_scripts": [
            "azul-smart-string-filter = azul_smart_string_filter.train.main:cli",
            "azul-smart-string-filter-server = azul_smart_string_filter.restapi.filter_strings:main",
        ]
    },
    use_scm_version=True,
    setup_requires=["setuptools_scm"],
    install_requires=[r.strip() for r in open_file("requirements.txt") if not r.startswith("#")],
)
