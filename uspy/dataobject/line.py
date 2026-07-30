"""
1D data containers (lines / spectra).
"""

# pylint: disable=abstract-method
from __future__ import annotations
from typing import Any, Callable, Union
from numbers import Number

import numpy as np
import scipy.interpolate
import scipy.integrate
import pandas as pd

from uspy.dataobject.base import DataObject, _ArithmeticMixin
from uspy.dataobject.deferred import materialize


class Line(_ArithmeticMixin, DataObject):
    """
    Base class for all 1D data.

    Parameters
    ----------
    source : Union[str, np.ndarray]
        When a str: path to .csv file with x y data as columns. Delimiter is guessed from the file
        When np.ndarray: 2d numpy array containing the x and y values either als rows or cols

    Attributes
    ----------
    x, y : np.ndarray
        Numpy arrays containing the x and y data of the line
    ydim, xdmin : str
        Dimension of the x and y values, e.g. time, energy ...

    Examples
    --------
    Initialization from .csv file

    .. code-block:: none

       example.csv:
       U in V,  I in A
       0.0,     0.0
       1.0,     2.0

    >>> line = Line("example.csv")
    >>> line.x, line.y
    array([0.0, 1.0]), array([0.0, 2.0])

    When reading from a .csv file the column header is interpreted as *xdim* and *ydim*. If the
    header has the form *A in B*, *A* is the dimension and *B* is the corresponding unit.

    >>> line.xdim, line.ydim
    U, I

    The x and y values are also accessible over their corresponding dimension:

    >>> line.U
    array([0.0, 1.0])

    Initialization from numpy array

    >>> line = Line(np.array([[0.0, 1.0], [0.0, 2.0]]))
    >>> line.x
    array([0.0, 1.0])
    >>> line.xdim, line.ydim
    x, y

    """

    # pylint: disable=no-member
    _data_keys = ("x", "y")
    _meta_defaults = {
        "ydim": "y",
        "xdim": "x",
        "color": "k",
        "label": None,
    }
    _unit_defaults = {"x": "a.u.", "y": "a.u."}
    _op_attr = "y"

    def __init__(self, source: Union[str, np.ndarray]) -> None:
        super().__init__(source)

    def __getattr__(self, attr: str) -> Any:
        # Any private name (not just the literal "_data"/"_meta") has to be handled by
        # super() to avoid recursion: `self._meta` below is itself a property that reads
        # `_meta_store` out of __dict__, and during __new__/unpickling that key may not
        # exist yet -- routing every private-name miss straight to the base class (which
        # raises AttributeError immediately for those) avoids ever touching `_meta_store`
        # before it's there.
        if attr.startswith("_"):
            return super().__getattr__(attr)
        if attr == self._meta["xdim"]:
            return materialize(self._data["x"])
        if attr == self._meta["ydim"]:
            return materialize(self._data["y"])

        return super().__getattr__(attr)

    def __setattr__(self, attr: str, value: Any) -> None:
        # See __getattr__ above: every private name, not just "_data"/"_meta" themselves.
        if attr.startswith("_"):
            return super().__setattr__(attr, value)

        if attr == self._meta["xdim"]:
            self._data["x"] = value
        elif attr == self._meta["ydim"]:
            self._data["y"] = value
        else:
            return super().__setattr__(attr, value)

    def parse(self, source: Union[str, np.ndarray]) -> dict[str, Any]:
        if isinstance(source, np.ndarray):
            if source.ndim == 2:
                if source.shape[0] == 2:
                    source = source.T
                df = pd.DataFrame(source)
                self._source = None

            else:
                raise ValueError("Not a Line")
        elif isinstance(source, str):
            df = pd.read_csv(source)

            if len(df.columns) != 2:
                raise ValueError("Not a Line")

            self._source = source

        x = df.to_numpy()[:, 0]
        y = df.to_numpy()[:, 1]

        xdim, ydim = list(df.columns)
        x_unit, y_unit = self._unit_defaults["x"], self._unit_defaults["y"]
        if (xdim, ydim) == (0, 1) or (xdim, ydim) == ("0", "1"):
            xdim = self._meta_defaults["xdim"]
            ydim = self._meta_defaults["ydim"]
        else:
            try:
                xdim, x_unit = xdim.split(" in ")
                ydim, y_unit = ydim.split(" in ")
            except ValueError:
                pass

        return {
            "x": x,
            "y": y,
            "xdim": xdim,
            "ydim": ydim,
            "x_unit": x_unit,
            "y_unit": y_unit,
        }

    @property
    def length(self) -> int:
        """Length of the x and y data."""
        return len(self.x)

    @property
    def area(self) -> float:
        "Area under the line integrated by simpson rule"
        return scipy.integrate.simpson(self.y, self.x)

    @property
    def dataframe(self) -> pd.DataFrame:
        """Pandas Dataframe of the lines x,y data"""
        return pd.DataFrame(
            data=np.stack([self.x, self.y], axis=1),
            columns=[
                f"{self.xdim} in {self._units['x']}",
                f"{self.ydim} in {self._units['y']}",
            ],
        )

    def interpolate(
        self,
        x_data: Union[Number, list[Number], np.ndarray, None] = None,
        order: Union[str, int] = "cubic",
        **kwargs,
    ) -> Union[np.ndarray, Callable]:
        """
        Interpolates the x,y data of the Line with a spline of order 'order'

        Parameters
        ----------
        x_data: Number, list of numbers or 1d numpy array or None
            x_values at which the interpolation will be evaluated
        order: str or int
            When a string must be either "linear", "quad" or "cubic".
            When an int must be 1 <= order <= 5
        kwargs:
            Additional keyword arguments are passed through to
            scipy.interpolate.InterpolatedUnivariateSpline

        Returns
        -------
        if x_data was not None:
            numpy array containing interpolated Numbers
        if x_data was None:
            the interpolation function itself
        """

        k = {"linear": 1, "quad": 2, "cubic": 3}
        if order not in k.keys():
            k[order] = order

        func = scipy.interpolate.InterpolatedUnivariateSpline(
            self.x, self.y, k=k[order], **kwargs
        )
        if x_data is None:
            return func
        return func(x_data)

    def integral(self) -> Line:
        """Returns the integrated values of a cubic spline evaluated at the x values of the line"""
        func = self.interpolate()
        y = np.array([func.integral(self.x[0], x) for x in self.x])
        return Line(np.array([self.x, y]))

    def derivative(self) -> Line:
        """Returns the derivative of a cubic spline evaluated at the x values of the line"""
        func = self.interpolate()
        y = np.array([func.derivative()(x) for x in self.x[1:]])
        return Line(np.array([self.x[1:], y]))

    def smooth(self, kernel: Union[int, np.ndarray]) -> Line:
        """
        Recieves a kernel and convolves the values of the line

        Parameters
        ----------
        kernel : np.ndarray, int
            Kernel that will be convolved with the line data.
            If ``int`` an equaly distributed kernel of size ``kernel`` will be applied.

        Returns
        -------
        Line
            smoothed line

        Examples
        --------
        Rolling average over three data points of line. Note that sum(kernel) should be 1.

        >>> kernel = np.array([1,1,1])/3
        >>> smoothed = line.smooth(kernel)

        Above code is equivalent to:

        >>> smoothed = line.smooth(3)

        The weights do not have to be the same. To put more weigh on the current point than on
        adjecent points:

        >>> kernel = np.array([1,2,1])/4
        >>> smoothed = line.smooth(kernel)
        """

        if isinstance(kernel, int):
            kernel = np.ones(kernel) / kernel

        line = self.copy()
        # pylint: disable=attribute-defined-outside-init
        line.x, line.y = self.x, np.convolve(self.y, kernel, mode="same")
        line.y[0], line.y[-1] = self.y[0], self.y[-1]

        return line

    def save(self, fname: str, **kwargs) -> Line:
        """
        Saves the line as .csv

        Parameters
        ----------
        fname: str
            Filename of the file to save.
            If it ends with .csv a csv file of the dataframe representation of the line is saved
            Otherwise the parent(dataobject) save method is called.
        **kwargs
            Only meaningful for an HDF5 target (.h5/.hdf5/.nxs); see `uspy.hdf5.write`.

        Returns
        -------
            None
        """
        if fname.lower().endswith(".csv") or fname.lower().endswith(".txt"):
            if kwargs:
                raise TypeError(
                    f"save() got unexpected keyword argument(s) {list(kwargs)!r}: these are "
                    "only accepted for an HDF5 target (.h5/.hdf5/.nxs)."
                )
            self.dataframe.to_csv(fname, index=False, header=True)
        elif fname.lower().endswith(".tsf"):  # This is for CasaXPS
            if kwargs:
                raise TypeError(
                    f"save() got unexpected keyword argument(s) {list(kwargs)!r}: these are "
                    "only accepted for an HDF5 target (.h5/.hdf5/.nxs)."
                )
            self.dataframe.to_csv(fname, index=False, header=False, sep="\t")
        else:
            super().save(fname, **kwargs)

        return self

    def is_compatible(self, other: Line) -> bool:
        try:
            assert super().is_compatible(other)
            assert self.length == other.length
            return True
        except AssertionError:
            return False

