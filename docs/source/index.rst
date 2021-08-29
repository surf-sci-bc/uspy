.. µSPY documentation master file, created by
   sphinx-quickstart on Sat Aug 28 21:57:53 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to µSPY's documentation!
================================

**µSPY** (micro-spy) provides and easy to use framework for analysing microscopy and microspectrocopy
data. It's core functionallity currently focuses on Low energy electron microscopy (LEEM) data, but 
is easily extandable, as long as you know how to parse your data files.

uSPY is completely written in Python and provides a methods that allow fast and reliable 
quantitative analysis of with only a few lines of code. Plotting of images (with overlaying 
metadata) is as easy as:

.. code-block:: python

   plot_img("/path/to/file")

µSPY was created with Jupyter in mind and is extremly powerful when used inside a Notebook. Because 
µSPYs interface is based on method calls, the analysis is always fully documented and completely
reproducable, underlying data, evaluation steps and results are bundled together, which makes time 
consuming reevaluation of data superfluous (looking at you Fiji). Working with µSPY is even fast
enough that you can analyse while measuring, using your Jupyter Notebook as a Labook. 


.. toctree::
   :maxdepth: 2
   :caption: Contents:

API Reference
-------------

.. toctree::
   :maxdepth: 3

   api/index




Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
