"""Setup script."""
# pylint: disable=invalid-name

from setuptools import setup, find_packages
from agfalta.version import __version__

# WARNING: seems like dependencies are not properly declared in opencv
# first, do:
# cd to this 'setup.py's directory
# $ python3 -m venv venv                        # create virtual environment
# OR $ source venv/bin/activate                 # or enter existing venv
# OR venv\Scripts\activate                      # enter existing venv on windows
# $ python3 -m pip install --upgrade pip        # sometimes necessary?
# $ python3 -m pip install scikit-build matplotlib cmake    # pyclustering has wrong dependencies
# linux only: $ sudo apt install python3-opencv
# linux only: $ cp /usr/lib/python3/dist-packages/cv* venv/lib/python3.8/site-packages/
# $ python3 -m pip install "opencv-python<4.0" "opencv-contrib-python<4.0"
# $ python3 -m pip install -e .


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
