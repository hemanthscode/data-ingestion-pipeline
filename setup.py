"""
Setup script for Data Ingestion Pipeline
"""
from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Read requirements
requirements = (this_directory / "requirements.txt").read_text().splitlines()
requirements = [req.strip() for req in requirements if req.strip() and not req.startswith('#')]

setup(
    name="data-ingestion-pipeline",
    version="1.0.0",
    author="Hemanth Sayimpu",
    author_email="hemanths7.dev@gmail.com",
    description="Production-ready data ingestion and preprocessing pipeline",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hemanthscode/data-ingestion-pipeline",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "pytest-cov>=4.1.0",
            "black>=23.12.1",
            "flake8>=6.1.0",
            "isort>=5.13.2",
            "mypy>=1.7.1",
        ],
        "docs": [
            "sphinx>=7.2.6",
            "sphinx-rtd-theme>=2.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "data-pipeline=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["config/*.yaml"],
    },
    keywords=[
        "data-pipeline",
        "data-cleaning",
        "data-preprocessing",
        "etl",
        "data-quality",
        "machine-learning",
        "data-analysis",
    ],
    project_urls={
        "Bug Reports": "https://github.com/hemanthscode/data-ingestion-pipeline/issues",
        "Source": "https://github.com/hemanthscode/data-ingestion-pipeline",
        "Documentation": "https://data-ingestion-pipeline.readthedocs.io",
    },
)