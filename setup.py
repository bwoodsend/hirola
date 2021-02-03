# -*- coding: utf-8 -*-
"""
"""

from setuptools import setup, find_packages
import runpy
from pathlib import Path

from cslug.building import (build_slugs, bdist_wheel, CSLUG_SUFFIX,
                            copy_requirements)

HERE = Path(__file__).resolve().parent

readme = (HERE / 'README.rst').read_text("utf-8")

setup(
    author="Brénainn Woodsend",
    author_email='bwoodsend@gmail.com',
    python_requires='>=3.6',
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    description="NumPy vectorized hash table for fast set and dict operations.",
    install_requires=copy_requirements(),
    extras_require={
        "test": [
            'pytest>=3', 'pytest-order', 'coverage', 'pytest-cov',
            'coverage-conditional-plugin'
        ]
    },
    license="MIT license",
    long_description=readme,
    package_data={
        "hoatzin": ["*" + CSLUG_SUFFIX, "*.json"],
    },
    keywords='NumPy Hash-table dict set',
    name='hoatzin',
    packages=find_packages(include=['hoatzin', 'hoatzin.*']),
    url='https://github.com/bwoodsend/hoatzin',
    version=runpy.run_path(HERE / "hoatzin/_version.py")["__version__"],
    zip_safe=False,
    cmdclass={
        "build": build_slugs("hoatzin._hash_table:slug"),
        "bdist_wheel": bdist_wheel,
    },
)
