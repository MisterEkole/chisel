from setuptools import setup, find_packages

setup(
    name="chisel",
    version="0.1.0",
    description="SFM and non-linear optimization pipeline",
    author="Mitterrand Ekole",
    packages=find_packages(), 
    python_requires=">=3.8",
)