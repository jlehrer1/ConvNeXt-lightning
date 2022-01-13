#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

requirements = [
    "pandas",
    "numpy",
    "torch",
    "pytorch-lightning",
    "lightning-bolts",
    "Pillow",
    "timm",
]

setup_requirements = requirements.copy()

test_requirements = [ ]

setup(
    author="Julian Lehrer",
    author_email='jmlehrer@ucsc.edu',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v2 (GPLv2)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="An implementation of the ConvNeXt architecture built on the PyTorch-Lightning API",
    install_requires=requirements.copy(),
    license="MIT license",
    long_description="",
    include_package_data=True,
    keywords='convnextpl',
    name='convnextpl',
    packages=find_packages(exclude=['tests']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/jlehrer1/ConvNeXt-lightning',
    version='0.0.1',
    zip_safe=False,
)
