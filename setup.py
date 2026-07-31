"""Setup script."""

# pylint: disable=invalid-name

from setuptools import setup, find_packages


setup(
    name="uspy",
    use_scm_version={
        "root": ".",
        "relative_to": __file__,
        "fallback_version": "0.0.0",
    },
    author="Simon Fischer, Lars Buß, Jon-Olaf Krisponeit",
    description="LEEM data analysis (and more...?)",
    packages=find_packages(include=["uspy", "uspy.*"]),
    classifiers=(
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
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
        #"scikit-video",
        "scikit-video @ git+https://github.com/scikit-video/scikit-video.git@v1.3.0", # >= 1.1.12 for compatibility with numpy
        "symmetrize",
        "imageio",
        "tifffile",
        # "pyclustering", This is broken
        "pyclustering @ git+https://github.com/KulikDM/pyclustering.git@Warning-Fix",  # This should fix it
        "natsort",
        "numba",
        "nltk",
        "opencv-python-headless",
        "kneed",
        "lmfit",
        "ipython",
        "h5py>=3.0",  # NeXus-flavoured HDF5 export/import (uspy.hdf5); >=3.0 for Dataset.asstr
        "json_tricks",
        "setuptools_scm",  # see uspy/version.py
        'tomli; python_version < "3.11"',  # tomllib is stdlib from 3.11 on
    ],
    package_data={"uspy.legacy.xps": ["rsf.db"]},
    python_requires=">=3.10",
    extras_require={"test": ["pytest", "pytest-cov"]},
)

### Additional packages
# dev suggestions: black, pylint
# for building and publishing: build, twine
