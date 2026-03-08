import pathlib
from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text()
DESCRIPTION = "scTenifoldpy"

PKGS = find_packages()
PKG_NAME = "scTenifoldpy"
PKG_VERSION = '0.1.3'

MAINTAINER = 'Yu-Te Lin'
MAINTAINER_EMAIL = 'qwerty239qwe@gmail.com'

PYTHON_REQUIRES = ">=3.7"
URL = "https://github.com/qwerty239qwe/scTenifoldpy"

LICENSE = "MIT"
CLFS = [
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.7",
]

# update the install requires
# let pip chooses the appropriate version of the dependencies based on the user's environment
INSTALL_REQUIRES = [
    "pandas>=1.2",
    "numpy>=1.20",
    "scipy>=1.6",
    "setuptools>=56",
    "typer>=0.4",
    "PyYAML>=5.4",
    "ray>=1.8",
    "scikit-learn>=0.24",
    "tensorly>=0.6",
    "requests>=2.26",
    "seaborn>=0.11",
    "matplotlib>=3.4",
    "networkx>=2.6",
    "scanpy>=1.7",
    "protobuf>=3.20,<5"
]
# This call to setup() does all the work
setup(
    name=PKG_NAME,
    version=PKG_VERSION,
    description=DESCRIPTION,
    long_description=README,
    long_description_content_type="text/markdown",
    url=URL,
    author=MAINTAINER,
    author_email=MAINTAINER_EMAIL,
    license=LICENSE,
    classifiers=CLFS,
    packages=PKGS,
    include_package_data=True,
    install_requires=INSTALL_REQUIRES,
    entry_points={
        "console_scripts": [
            "scTenifold=scTenifold.__main__:app",
        ]
    },
)