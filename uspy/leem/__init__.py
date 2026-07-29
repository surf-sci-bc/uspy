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

from uspy.leem.base import LEEMImg#, LEEMStack
#from uspy.leem.plotting import (
#     plot_img,
#     plot_mov, plot_movie, plot_meta, make_video,
#     print_meta,
#     get_intensity, plot_rois,
#     plot_intensity, plot_iv, plot_intensity_img, plot_iv_img,
#     plot_rsm
# )
# from uspy.leem.processing import calculate_dose, ROI, Profile


def __getattr__(name: str) -> str:
    """`uspy.leem.LEEMDIR`/`DATADIR` used to be re-exported here from `uspy` (though
    never actually used within this module); forwarded lazily so `import uspy.leem`
    doesn't read a uspy.toml just to populate an unused name."""
    if name in ("LEEMDIR", "DATADIR"):
        import uspy

        return getattr(uspy, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
