from setuptools import setup, find_packages

setup(
    name="openmatch",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch",
        "transformers",
        "datasets",
        "pytrec_eval",
        "Pillow",
    ],
)