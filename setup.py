from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

requirements = []
with open('requirements.txt') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="twitter-layoffs-analysis",
    version="1.0.0",
    author="Angad Bawa",
    long_description_content_type="text/markdown",
    url="https://github.com/angadbawa/Twitter-Layoffs-Analysis-2024",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=23.0",
            "flake8>=6.0",
            "mypy>=1.0",
        ],
        "jupyter": [
            "jupyter>=1.0",
            "jupyterlab>=3.0",
            "ipywidgets>=8.0",
        ],
        "viz": [
            "plotly>=5.0",
            "wordcloud>=1.9",
            "seaborn>=0.12",
        ]
    },
    entry_points={
        "console_scripts": [
            "twitter-analysis=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "twitter_analysis": ["config/*.yaml"],
    },
    zip_safe=False,
)
