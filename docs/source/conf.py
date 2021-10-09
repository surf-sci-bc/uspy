# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys
import inspect
sys.path.insert(0, os.path.abspath('../../uspy'))
sys.path.insert(0, os.path.abspath('../../uspy/leem'))


# -- Project information -----------------------------------------------------

project = 'µSPY'
copyright = '2021, Simon Fischer, Lars Buß'
author = 'Simon Fischer, Lars Buß'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.napoleon',
    'sphinx.ext.autodoc',
    'sphinx_copybutton',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# Autodoc settings
autoclass_content = "both"
add_module_names = False
autodoc_typehints = "none"


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'furo'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_logo = '_static/logo_uspy.jpg'

def change_new_to_init(app, obj, bound_method):
    """
    When a __new__ method is defined for a class, Sphinx automatically uses the Signature of the
    __new__ method as signature for the class. Because __new__ normally does not specify the
    contructor arguments, the signature of __new__ has to be replaced with the signature of __init__
    """

    if obj.__qualname__ == "DataObject.__new__":
        import uspy.dataobject as dataobject
        init = dataobject.DataObject.__init__
        new_sig = inspect.signature(init).replace(return_annotation=inspect.Signature.empty)
        obj.__text_signature__ = str(new_sig)

def setup(app):
    app.connect("autodoc-before-process-signature", change_new_to_init)
