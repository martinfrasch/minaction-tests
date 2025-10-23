"""
Setup configuration for minAction-LLM-Physics-Tests
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="minaction-llm-physics-tests",
    version="1.0.0",
    author="Martin G. Frasch",
    author_email="mfrasch@uw.edu",
    description="Testing whether mathematical language models understand physical selection principles",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/martinfrasch/minAction-LLM-physics-tests",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "minaction-test=scripts.run_tests:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.json"],
        "prompts": ["**/*.txt"],
        "results": ["**/*.json", "**/*.txt"],
    },
)
