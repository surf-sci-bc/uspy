"""
Composite data containers derived from an ImageStack/Image plus a uspy.roi selection.

These are the only dataobject types coupled to uspy.roi -- they take *tuples of upstream
objects* (stack, ROI, axis name, ...) as their source, so their parse() does extraction/
aggregation rather than file I/O.
"""

# pylint: disable=abstract-method
from __future__ import annotations
from typing import Any, Sequence, Union

import numpy as np
import pandas as pd
import scipy.optimize

import uspy.roi as roi
from uspy.dataobject.image import Image, ImageStack
from uspy.dataobject.line import Line


class IntensityLine(Line):
    """
    A Class to extract Intensities from an ImageStack

    Unlike Line from which it inherits it takes an ImagseStack, a ROI and a String representing the
    dimension along which the itensities in the stack should be extracted, e.g. time, energy ...
    For each image the the mean value of intensites inside the ROI is extracted as y-values together
    with the dimensional value of the image (time, energy, ...) as the lines x-axes.
    """

    def __init__(self, stack: ImageStack, roi_: roi.ROI, xaxis: str) -> None:
        """
        Parameters
        ----------
        stack : ImageStack
            ImageStack from which the intensities will be extracted
        roi_ : roi.ROI
            ROI from within the intensities will be extracted in each image,
        xaxis : str
            a string indicates the axis of the dimension along the intensites shall be extracted,
            e.g. time, energy ...
        """
        super().__init__((stack, roi_, xaxis))

    def parse(self, source: tuple[ImageStack, roi.ROI, str]) -> dict[str, Any]:
        """Applies a ROI to an Imagestack and extracts mean of ROI from each image

        Parameters
        ----------
        source : tuple[ImageStack, roi.ROI, str]
            Must be a Tuple of

            1. an ImageStack from which the intensities will be extracted,
            2. a ROI from within the intensities will be extracted in each image,
            3. a string indicates the axis of the dimension along the intensites shall be \
                extracted, e.g. time, energy ...

        Returns
        -------
        dict
            a dict containing the x,y-values of the line, the stack, the ROI, the x- and
            y-dimensions and the unit of the x axis extracted from the images. The y-dimension is
            "intensity" by default with unit "a.u."
        """

        stack, roi_, xaxis = source
        self._source = source
        y = []
        x = []
        for img in stack:
            masked_img = roi_.apply(img)
            y.append(np.mean(masked_img.image))
            x.append(getattr(img, xaxis))

        return {
            "x": np.array(x),
            "y": np.array(y),
            "stack": stack,
            "roi": roi_,
            "xdim": xaxis,
            "ydim": "intensity",
            "x_unit": stack[0].unit.get(xaxis, ""),
            "label": roi_.label,
        }


class StitchedLine(Line):
    """
    Class for combining IntensityLines from multiple ImageStacks.

    Works like IntensityLine, but accepts multiple ImageStacks and the same amount of ROIs. Every
    stack is mapped to a ROI (in the order that they are given, e.g. first stack to first ROI) and
    the intensity profile is extracted. If the x-values of the extracted Lines are overlapping, the
    Lines are scaled to matched each other. The stacks may be sorted by their first xaxis value
    before extraction.

    Attributes
    ----------
    x, y : np.ndarray
        x and y values of stitched lines
    lines : list[IntensityLine]
        list of single IntensityLines that are used for stitching
    xdim, ydim : str
        string representing the dimension of the x and y values, default for y is "intensity"
    """

    def __init__(
        self,
        stacks: Union[Sequence[ImageStack], ImageStack],
        rois: Union[Sequence[roi.ROI], roi.ROI],
        xaxis: str,
    ) -> None:
        """
        Parameters
        ----------
        stacks : Sequence[ImageStack] or ImageStack
            ImageStacks, which will be used for Line extraction
        rois : Sequence[roi.ROI] or roi.ROI
            ROIs used for Line extraction. Must be of the same length as ``stacks``. If stacks is a
            single stack, ROI has to be a single ROI
        xaxis:
            a string indicates the axis of the dimension along the intensites shall be extracted,
            e.g. time, energy ...

        Raises
        ------
        ValueError
            When len(stacks) != len(rois)
        """
        if isinstance(stacks, ImageStack):
            stacks = [stacks]
            rois = [rois]

        super().__init__((stacks, rois, xaxis))

    def parse(self, source: Sequence[Sequence, Sequence, str]) -> dict[str, Any]:
        """Extract IntensityLines from Stacks and handle their stitching.

        Parameters
        ----------
        source : Sequence[Sequence, Sequence, str]
            List or tuple containing

            1. A list or tuple of ImageStacks
            2. A list or tuple of ROIs with the same length as the tuple of ImageStacks
            3. A str representing the xaxis along which the intensity is extracted

        Returns
        -------
        dict[str, Any]
            dict containing all metadata and data, including rois and stacks

        Raises
        ------
        ValueError
            When len(stacks) != len(rois)
        """

        stacks, rois, xaxis = source

        if len(stacks) != len(rois):
            raise ValueError(f"Cannot map {len(rois)} to {len(stacks)}")

        lines = [
            IntensityLine(stack, roi, xaxis)
            for stack, roi in sorted(
                zip(stacks, rois), key=lambda x: getattr(x[0][0], xaxis)
            )
        ]

        if len(lines) == 1:
            x, y = lines[0].x, lines[0].y
        else:
            x, y = self._stitch_curves(lines)

        self._source = source

        return {
            "x": x,
            "y": y,
            "lines": lines,
            "xdim": xaxis,
            "ydim": "intensity",
            "x_unit": stacks[0][0].unit.get(xaxis, ""),
        }

    def _stitch_curves(
        self, lines: Sequence[IntensityLine]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Combine x,y values to mulitple lines to one pair of x,y values

        Parameters
        ----------
        lines : Sequence[IntensityLine]
            List or tuple of Lines to be stitched together

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            tuple containing the x and y values of the stitched curve
        """
        coeffs = [1]

        for curr_line, next_line in zip(lines, lines[1:]):
            # calculate overlap in x values between lines, assuming ordered values:

            start_x = next_line.x[0]
            end_x = curr_line.x[-1]

            if start_x > end_x:
                print("No Overlap detected.")
                coeffs.append(1)
            elif start_x == end_x:
                print("One Point overlap")
                coeffs.append(next_line.y[0] / curr_line.y[-1])
            else:
                curr_line_points = np.array(
                    [
                        (x, y)
                        for x, y in zip(curr_line.x, curr_line.y)
                        if start_x <= x <= end_x
                    ]
                )
                next_spline = next_line.interpolate(order="cubic")

                # pylint: disable=cell-var-from-loop
                func = lambda x, a: a * next_spline(x)
                # pylint: disable=unbalanced-tuple-unpacking
                popt, _ = scipy.optimize.curve_fit(
                    func, curr_line_points[:, 0], curr_line_points[:, 1]
                )

                coeffs.append(popt)

        x_data = np.array([])
        y_data = np.array([])

        for coeff, line in zip(coeffs, lines):
            x_data = np.append(x_data, line.x)
            y_data = np.append(y_data, coeff * line.y)

        # sort values and average over same x values
        data = np.array([x_data, y_data]).T
        df = pd.DataFrame(data).groupby(0).mean()

        x = df.index.values
        y = df.values.flatten()

        return x, y


class ProfileLine(Line):
    def __init__(self, img, profile) -> None:
        super().__init__((img, profile))

    def parse(self, source: Sequence[Image, roi.Profile]) -> dict[str, Any]:
        img, profile = source

        y = profile.apply(img)
        x = np.linspace(0, len(y), len(y))

        self._source = source

        return {
            "x": x,
            "y": y,
            "xdim": "pixel",
            "ydim": "intensity",
            "x_unit": "px",
            "img": img,
            "profile": profile,
        }


class Waterfall(Image):
    def __init__(self, stack: ImageStack, profile: roi.Profile, yaxis: str) -> None:
        super().__init__((stack, profile, yaxis))

    def parse(self, source: Sequence[ImageStack, roi.Profile, str]) -> dict[str, Any]:
        stack, profile, yaxis = source

        wf = np.stack([profile.apply(img) for img in stack])
        yval = getattr(stack, yaxis)

        self._source = source

        return {
            "image": wf,
            "x": ProfileLine(stack[0], profile).x,
            "y": yval,
            "stack": stack,
            "profile": profile,
            "yaxis": yaxis,
            "height": wf.shape[0],
            "width": wf.shape[1],
        }
