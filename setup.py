"""Setup script."""
# pylint: disable=invalid-name

from setuptools import setup, find_packages
from agfalta.version import __version__


setup(
    name="agfalta",
    version=__version__,
    author="Simon Fischer, Lars Buß, Jon-Olaf Krisponeit",
    description="LEEM data analysis (and more...?)",
    packages=find_packages(exclude=["testdata", "doc", "tests", ".git"]),
    classifiers=(
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Physics"
    ),
    # license="GPLv3",
    keywords="physics LEEM microscopy spectroscopy",
    install_requires=[
        "pytest",
        "scikit-build",
        "cmake",
        "matplotlib",
        "numpy",
        "scikit-learn",
        "scikit-image",
        "kneed",
        "pyclustering",
        "nltk",
        "opencv-python<4.0",
        "opencv-contrib-python<4.0",
    ],
    python_requires="~=3.6",
    package_data={
        "agfalta": [
            "testdata/*"
        ]
    }
)
