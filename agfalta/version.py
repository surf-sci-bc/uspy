"""Retrieve version either live from git or from the distribution."""

try:
    from setuptools_scm import get_version
    __version__ = get_version(root="..", relative_to=__file__)
except (ImportError, LookupError):
    from pkg_resources import get_distribution
    __version__ = get_distribution(__package__).version
