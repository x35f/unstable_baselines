from setuptools import find_packages
from setuptools import setup

setup(
    name='unstable_baselines',
    auther='lamda-5',
    packages=find_packages(),
    python_requires='>=3.7',
    package_data={
        # include default config files and env data files
        "": ["*.yaml", "*.xml", "*.json"],
    }
)