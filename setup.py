from setuptools import find_packages
from setuptools import setup

setup(
    name='unstable_baselines',
    auther='lamda-5',
    # packages=find_packages(),
    packages=[package for package in find_packages() if package.startswith("unstable_baselines")], 
    python_requires='>=3.7',
    package_data={
        # include default config files and env data files
        "": ["*.yaml", "*.xml", "*.json"],
    }
)