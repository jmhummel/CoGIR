from setuptools import setup, find_packages

# Read requirements from requirements-controlnet.txt
with open("requirements-controlnet.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="ControlNet",
    version="1.0",
    packages=find_packages(),
    install_requires=requirements,  # Use requirements.txt for dependencies
)