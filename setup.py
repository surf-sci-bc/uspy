"""Setup script."""
# pylint: disable=invalid-name

from setuptools import setup, find_packages


setup(
    name="uspy",
    use_scm_version={
        "root": ".",
        "relative_to": __file__,
        "fallback_version": "NOT-INSTALLED-VERSION",
    },
    author="Simon Fischer, Lars Buß, Jon-Olaf Krisponeit",
    description="LEEM data analysis (and more...?)",
    packages=find_packages(include=["uspy", "uspy.*"]),
    classifiers=(
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Physics",
    ),
    license="MIT",
    keywords="physics LEEM microscopy spectroscopy",
    setup_requires=["setuptools_scm"],
    install_requires=[
        "numpy",
        "deepdiff",
        "matplotlib",
        "pandas",
        "scikit-image",
        "scikit-learn",
        "scikit-video",
        "symmetrize",
        "imageio",
        "tifffile",
        "pyclustering",
        "natsort",
        "nltk",
        "opencv-python-headless",
        "kneed",
        "lmfit",
        "ipython",
        "netCDF4",
        "json_tricks",
        "setuptools_scm",  # see uspy/version.py
    ],
    package_data={"uspy.xps": ["rsf.db"]},
    python_requires="~=3.6",
    tests_require=["pytest", "pytest-cov"],
)

### Additional packages
# dev suggestions: black, pylint
# for building and publishing: build, twine
