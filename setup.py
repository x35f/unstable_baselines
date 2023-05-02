from setuptools import find_packages
from setuptools import setup

with open("./VERSION.txt", "r") as fp:
    __VERSION__ = fp.readline().strip()

setup(
    name='unstable_baselines',
    author='lamda5-z',
    # packages=find_packages(),
    packages=[package for package in find_packages() if package.startswith("unstable_baselines")], 
    python_requires='>=3.7',
    package_data={
        # include default config files and env data files
        "": ["*.yaml", "*.xml", "*.json"],
    }, 
    license="MIT", 
    version=__VERSION__, 
    url="https://github.com/x35f/unstable_baselines", 
)
