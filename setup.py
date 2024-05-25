import os.path
from setuptools import setup, find_packages
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

setup(
    name="autosyn",
    version="1.0",
    author="all",
    author_email="",
    description="Stay true",

    packages=find_packages()
)