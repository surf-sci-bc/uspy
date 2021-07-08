"""
Utilities for LEEM data inspection and analysis.
Available functions:
- plot_img(img)
- plot_mov(stack)
- plot_meta(stack)
- print_meta(stack_or_img)
- plot_intensity(stack, ROI)
- plot_intensity_img(stack, ROI)
- calculate_dose(stack)

And possibly others. Use help(function) to find out more
about each function or look at the demo files.
"""

from agfalta import LEEMDIR

from agfalta.leem.base import LEEMImg, LEEMStack
from agfalta.leem.plotting import (
    plot_img,
    plot_mov, plot_movie, plot_meta, make_video,
    print_meta,
    get_intensity, plot_rois,
    plot_intensity, plot_iv, plot_intensity_img, plot_iv_img,
    plot_rsm
)
from agfalta.leem.processing import calculate_dose, ROI, Profile
