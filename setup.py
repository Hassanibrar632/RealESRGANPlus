from setuptools import setup, find_packages
import os

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name='RealESRGANPlus',
    py_modules=["RealESRGANPlus"],
    version='1.0.0',
    description='Real-ESRGAN: Practical Algorithms for General Image Restoration. This is an unofficial implementation of Real-ESRGANPlus, a PyTorch-based image restoration model.',
    author='M. Hassan Ibrar',
    author_email='hassanibrar632@gmail.com',
    url='https://github.com/Hassanibrar632/RealESRGANPlus',
    packages=find_packages(include=['RealESRGANPlus']),
    install_requires=requirements,
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)