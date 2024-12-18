"""Spectrum class represents spectrum data."""
# pylint: disable=too-many-instance-attributes
# pylint: disable=missing-docstring
# pylint: disable=invalid-name

import re
from typing import Any, Union

import numpy as np
import matplotlib.pyplot as plt
from lmfit import Parameters, report_fit
from lmfit.models import ConstantModel  # , PseudoVoigtModel
import lmfit.models as lmmodels

from uspy.xps import processing, models
from uspy.dataobject import Line


class XPSSpectrum(Line):
    """
    Holds data from one single spectrum.
    """

    background_types = (None, "linear", "shirley", "tougaard")
    norm_types = (None, "highest", "high_energy", "low_energy", "manual", "energy")
    _meta_defaults = {
        "ydim": "intensity",
        "xdim": "energy",
        "color": "k",
        "raw_is_binding": False,  # False if kinetic energy
        "sweeps": np.nan,
        "dwelltime": np.nan,
        "pass_energy": np.nan,
    }
    _unit_defaults = {"x": "eV", "y": "a.u."}

    def __init__(self, source):
        super().__init__(source)
        # self.photon_energy = photon_energy
        # self.binding_energy = binding_energy

        # if not binding_energy and photon_energy:
        #    self.binding_energy = True
        #    self.x = photon_energy - self.x + energy_calibration

        # self.path = path
        # self._meta = {}
        # self._energy = None
        # self._intensity = None

        # self.parse_txt(idx)

        # if self._meta["energy_scale"] == "kinetic":
        #     print("energy scale is not binding energy, trying to convert...")
        #     self._energy = self._meta["photon_energy"] - self._energy
        # if len(self._energy) != len(self._intensity) or self._energy.ndim != 1:
        #     raise ValueError("energy and intensity array sizes differ")
        # self._energy, self._intensity = processing.make_increasing(
        #     self._energy, self._intensity
        # )
        # self._energy, self._intensity = processing.make_equidistant(
        #     self._energy, self._intensity
        # )

        self._background = np.zeros_like(self.energy)
        self._background_type = None

        self.energy_calibration = 0  # correction to binding energy
        self.photon_energy = 0

        self._background_bounds = np.array([])

        self._raw_energy = np.copy(self.x)
        self._raw_intensity = np.copy(self.y)

        self._norm_divisor = 1.0
        self._norm_type = None

        # Fit related
        # self.params = Parameters()
        # self._peaks = []

    def parse(self, source: Union[str, np.ndarray, Line]) -> dict[str, Any]:
        if isinstance(source, Line):
            self._source = source
            return {
                "x": source.x,
                "y": source.y,
            }
        else:
            super().parse(source)

    # def parse_txt(self, idx):
    #     try:
    #         specdict = io.parse_spectrum_file(self.path)[idx]
    #     except FileNotFoundError as e:
    #         raise ValueError(
    #             f"{self.path} does not exist or has wrong file format"
    #         ) from e
    #     try:
    #         self._energy = specdict.pop("energy")
    #         self._intensity = specdict.pop("intensity")
    #     except KeyError as e:
    #         raise ValueError(f"{self.path} cannot be parsed correctly") from e
    #     self._meta = self._defaults.copy()
    #     self._meta.update(specdict)

    def __getattr__(self, attr):

        """
        Spectrum inherits from line, so calling self.x and self.energy should return the same array
        However, XPS spectra have different kinds of energys e.g raw energies, corrected energies,
        binding energies... , so energy (and intensity) are modified by a property. Properties are
        looked up before __getattr__ so this is valid. However, to hold the degenarcy of self.x ==
        self.energy, __getattr__ has to be directed to the properties when asking for x and y.
        """

        if attr in ("_data", "_meta"):
            return super().__getattr__(attr)
        if attr == "x":
            return self.energy
        if attr == "y":
            return self.intensity
        return super().__getattr__(attr)

    # def __setattr__(self, attr, value):
    #     if hasattr(self, "_meta") and attr in self._meta:
    #         self._meta[attr] = value
    #     else:
    #         super().__setattr__(attr, value)

    # Energy-related
    @property
    def energy(self):
        """Energy is binding energy"""
        if self.raw_is_binding is True:
            return self._raw_energy + self.energy_calibration
        return self.photon_energy - (self._raw_energy + self.energy_calibration)

    @property
    def kinetic_energy(self):
        if self.raw_is_binding is False:
            return self._raw_energy + self.energy_calibration
        return self.photon_energy - (self._raw_energy + self.energy_calibration)

    # Intensity-related
    @property
    def intensity(self):
        return self._raw_intensity / self._norm_divisor

    def intensity_at_E(self, energy):
        return processing.intensity_at_energy(self.energy, self.intensity, energy)

    def norm(self, norm_type="manual", value=1.0):
        if norm_type not in self.norm_types:
            raise ValueError(f"Invalid normalization type '{norm_type}'")
        if norm_type is None:
            self._norm_divisor = 1.0
        elif norm_type == "highest":
            self._norm_divisor = self.intensity.max()
        elif norm_type == "manual":
            self._norm_divisor = value
        elif norm_type == "energy":
            self._norm_divisor = 1.0
            self._norm_divisor = self.intensity_at_E(value)
        else:
            span = self.intensity.max() - self.intensity.min()
            if norm_type == "low_energy":
                idx = np.argmax(
                    np.abs(self.intensity - self.intensity[0]) > span * 0.05
                )
                self._norm_divisor = self.intensity[:idx:].mean()
            elif norm_type == "high_energy":
                idx = np.argmax(
                    np.abs(self.intensity[::-1] - self.intensity[-1]) > span * 0.05
                )
                self._norm_divisor = self.intensity[-1:idx:-1].mean()
        self._norm_type = norm_type

    @property
    def norm_type(self):
        return self._norm_type

    @property
    def norm_divisor(self):
        return self._norm_divisor

    @norm_divisor.setter
    def norm_divisor(self, value):
        self._norm_type = "manual"
        self._norm_divisor = value

    # Background-related
    @property
    def background(self):
        return self._background / self._norm_divisor

    def background_at_E(self, energy):
        return processing.intensity_at_energy(self.energy, self.background, energy)

    @property
    def background_type(self):
        return self._background_type

    def calculate_background(self, type_="shirley", bounds=None):
        if bounds is None:
            self._background_bounds = np.array(
                [np.amin(self.energy), np.amax(self.energy)]
            )
        else:
            if len(bounds) % 2 != 0:
                raise ValueError("Background bounds must be pairwise.")
            self._background_bounds = bounds

        # self._background_bounds = np.sort(bounds) - self.energy_calibration
        # self._background_bounds = self._background_bounds.clip(
        #    self._energy.min(), self._energy.max()
        # )

        if type_ not in self.background_types:
            raise ValueError(f"Background type {type_} not valid")
        self._background = processing.calculate_background(
            type_, self._background_bounds, self.energy, self.intensity
        )
        # self._background_type = value

    @property
    def background_bounds(self):
        return self._background_bounds  # + self.energy_calibration

    # @background_bounds.setter
    # def background_bounds(self, value):
    #     """Only even-length numeral sequence-types are valid."""
    #     if len(value) % 2 != 0:
    #         raise ValueError("Background bounds must be pairwise.")
    #     self._background_bounds = np.sort(np.array(value)) - self._energy_calibration
    #     self._background_bounds = self._background_bounds.clip(
    #         self._energy.min(), self._energy.max()
    #     )
    #     self._background = processing.calculate_background(
    #         self.background_type, self.background_bounds, self.energy, self._intensity
    #     )


class ModeledSpectrum(XPSSpectrum):
    """Holds information on the Fit and provides methods for fitting."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.params = Parameters()
        self._peaks = []
        self._result = None

    @property
    def peaks(self):
        """Returns peaks."""
        return self._peaks.copy()

    @property
    def result(self):
        """Returns peaks."""
        return self._result

    @property
    def fit_intensity(self):
        """Returns fit result on whole energy range."""
        with processing.IgnoreUnderflow():
            fit = self.model.eval(params=self.params, x=self.energy)
        try:
            if fit.shape == self.energy.shape:
                return fit
        except AttributeError:
            pass
        return np.zeros_like(self.energy)

    def fit_intensity_at_E(self, energy):
        """Returns model intensity at given energy."""
        energy = np.array([energy])
        with processing.IgnoreUnderflow():
            fit = self.model.eval(params=self.params, x=energy)
        return fit

    @property
    def residual(self):
        """Returns the fit residual."""
        return self.intensity - self.background - self.fit_intensity

    @property
    def model(self):
        """Returns the sum of all peak models."""
        # model = ConstantModel(prefix="BASE_")
        # model.set_param_hint("c", vary=False, value=0)
        # self.params += model.make_params()
        # model = None

        model = self._peaks[0].model

        for peak in self._peaks[1:]:
            model += peak.model
        return model

    @property
    def report(self):
        """Returns the report of the fit"""
        report_fit(self.params)

    def do_fit(self):
        """Returns the fitted cps values."""
        with processing.IgnoreUnderflow():
            result = self.model.fit(
                self.intensity - self.background,
                self.params,
                x=self.energy,
            )
        self.params.update(result.params)
        self._result = result
        return result

    def eval(self):
        return self.model.eval(
            self.params,
            x=self.energy,
        )

    def plot_eval(self, ax=None):
        if not ax:
            _, ax = plt.subplots()
        ax.plot(self.energy, self.eval(), label="eval")
        ax.plot(self.energy, self.intensity - self.background, label="data")
        ax.legend()
        # plt.show()

    # def add_peak(self, name, **kwargs):
    #     """
    #     Add a peak with given parameters. Valid parameters:
    #     area, fwhm, position and model specific parameters:
    #         PseudoVoigt: fraction, gausswidth
    #     """
    #     if name in [peak.name for peak in self._peaks]:
    #         # raise ValueError("Peak already exists")
    #         self.remove_peak(name)
    #     if "fwhm" not in kwargs and "height" in kwargs and "angle" in kwargs:
    #         kwargs["fwhm"] = models.pah2fwhm(
    #             kwargs["position"], kwargs["angle"], kwargs["height"], kwargs["shape"]
    #         )
    #     if "area" not in kwargs and "height" in kwargs:
    #         kwargs["area"] = models.pah2area(
    #             kwargs["position"], kwargs["angle"], kwargs["height"], kwargs["shape"]
    #         )

    #     kwargs.pop("angle", None)
    #     kwargs.pop("height", None)
    #     peak = Peak(name, self, **kwargs)
    #     self._peaks.append(peak)
    #     return peak

    def add_peak(self, peak):
        if peak.name in [p.name for p in self._peaks]:
            print(f"Updating Peak {peak.name}")
            self.remove_peak(next(filter(lambda x: x.name == peak.name, self._peaks)))
        self._peaks.append(peak)
        self.params.update(peak.params)

    def remove_peak(self, peak):
        """Remove a peak specified by its name."""
        if isinstance(peak, str):
            peak = next(filter(lambda x: x.name == peak, self._peaks))
        # remove paramerts that belong to peak
        pars_to_del = [
            # par for par in self.params if re.match(rf"{peak.name}_[a-z]+", par)
            par
            for par in self.params
            if par in peak.params
        ]
        for par in pars_to_del:
            self.params.pop(par)

        # peak.clear_params()
        self._peaks.remove(peak)
        # self.params += peak.params

    def set_constraints(self, param_alias, **new):
        """Sets a constraint for param. Valid keys:
        value, vary, min, max, expr"""
        try:
            param = self.get_param(param_alias)
        except ValueError:
            return

        old = self.get_constraints(param_alias)
        print(old)
        # enforce min=0 for all parameters except position:
        if (
            "min" not in new
            or old["min"] < 0
            and (new["min"] is None or new["min"] < 0)
            and param_alias != "position"
        ):
            new["min"] = 0
        # return if only None values are new
        if all(v is None for v in dict(set(new.items()) - set(old.items()))):
            return

        # if new["expr"]:
        #    new["expr"] = self.relation2expr(new["expr"], param_alias)

        try:
            param.set(**new)
            self.params.valuesdict()
        except (SyntaxError, NameError, TypeError):
            old["expr"] = ""
            param.set(**old)
            print(f"Invalid expression '{new['expr']}'")

    def get_constraints(self, param_alias):
        """Returns a string containing min/max or expr."""
        try:
            param = self.get_param(param_alias)
        except ValueError:
            return self._default_constraints
        constraints = {
            "value": param.value,
            "min": param.min,
            "max": param.max,
            "vary": param.vary,
            "expr": param.expr,
        }
        return constraints

    def add_param(self, *args, **kwargs):
        self.params.add(*args, **kwargs)

    def remove_param(self, par):
        self.params.pop(par)

    def get_param(self, param_alias, use_alias=False):
        """Shortcut for getting the Parameter object by the param alias."""
        if use_alias:
            aliases = {**self._default_aliases, **self.param_aliases}
            param_name = aliases.get(param_alias, param_alias)
        else:
            param_name = param_alias
        if param_name is None:
            raise ValueError(
                f"model '{self.shape}' does not support Parameter '{param_alias}'"
            )
        # return self.params[f"{self.name}_{param_name}"]
        return self.params[f"{param_name}"]


class Peak:
    """
    Provides read access to peak parameters and provides methods to
    constrain them.
    Whereever possible, parameter "aliases" are used. Independent of
    model, every peak should have:
        area
        fwhm
        position
    and optionally:
        alpha
    in the aliases, each of these properties are mapped to "real" parameters
    in a way that they all act similar.
    This ensures a consistent API.
    """

    _default_aliases = {
        "alpha": None,
    }
    _defaults = {
        "alpha": 0.5,
    }
    _default_constraints = {
        "value": None,
        "vary": True,
        "min": 0,
        "max": np.inf,
        "expr": "",
    }
    shapes = ["PseudoVoigt", "DoniachSunjic", "Voigt"]

    def __init__(
        self,
        name,
        # spectrum,
        # area=None,
        # fwhm=None,
        # position=None,
        # alpha=None,
        shape="Voigt",
        **kwargs,
    ):
        # pylint: disable=too-many-arguments
        super().__init__()
        self._name = name
        # self.spectrum = spectrum
        # self.params = spectrum.params
        self.params = Parameters()
        self._shape = shape
        self.label = f"Peak {name}"
        # if None in (area, fwhm, position):
        #    raise ValueError("Required attribute(s) missing")

        self._model = None
        # self.initialize_model(area, fwhm, position, alpha)
        self.initialize_model(**kwargs)

    @property
    def name(self):
        return self._name

    @property
    def model(self):
        return self._model

    # @property
    # def intensity(self):
    #     with processing.IgnoreUnderflow():
    #         intensity = self._model.eval(params=self.params, x=self.spectrum.energy)
    #     return intensity

    # def intensity_at_E(self, energy):
    #     with processing.IgnoreUnderflow():
    #         intensity = self._model.eval(params=self.params, x=energy)
    #     return intensity

    # def initialize_model(self, area, fwhm, position, alpha):
    def initialize_model(self, **kwargs):

        self.clear_params()

        guess = kwargs.pop("guess", False)

        if self._shape == "PseudoVoigt":
            # self.param_aliases["alpha"] = "fraction"
            # if alpha is None:
            #    alpha = 0.5
            # self._model = models.PseudoVoigtModel(prefix=f"{self.name}_")
            # self._model.set_param_hint("fraction", min=0, value=alpha)
            self._model = lmmodels.PseudoVoigtModel(prefix=f"{self.name}_")
        elif self._shape == "DS":

            self._model = lmmodels.DoniachModel(prefix=f"{self.name}_")
            # self.params = self._model.make_params()
            # self.get_param("amplitude").set(value=kwargs["amplitude"], min=0, vary=True)
            # self.get_param("center").set(value=kwargs["center"], min=0, vary=True)
            # self.get_param("gamma").set(value=kwargs["gamma"], min=0, vary=True)
            # self.get_param("sigma").set(value=kwargs["sigma"], min=0, vary=True)

            # self.param_aliases["alpha"] = "asym"
            # if alpha is None:
            #    alpha = 0.1
            # self._model = models.DoniachSunjicModel(prefix=f"{self.name}_")
            # self._model.set_param_hint("asym", min=0, value=alpha)

        elif self._shape == "Voigt":
            # self.param_aliases["alpha"] = "fwhm_l"
            # if alpha is None:
            #    alpha = 0.5
            self._model = lmmodels.VoigtModel(prefix=f"{self.name}_")
            self._model.set_param_hint("gamma", value=0.5, expr=None)
            # self.params = self._model.make_params()
            # self.get_param("amplitude").set(value=kwargs["amplitude"], min=0, vary=True)
            # self.get_param("center").set(value=kwargs["center"], min=0, vary=True)
            # self.get_param("gamma").set(value=kwargs["gamma"], min=0, vary=True)
            # self.get_param("sigma").set(value=kwargs["sigma"], min=0, vary=True)
            # self._model.set_param_hint()
            # self._model.set_param_hint("fwhm_l", min=0, value=alpha)
        elif self._shape == "Gaussian":
            self._model = lmmodels.GaussianModel(prefix=f"{self.name}_")
        elif self._shape == "Lorentzian":
            self._model = lmmodels.LorentzianModel(prefix=f"{self.name}_")
        elif self._shape == "SkewedVoigt":
            self._model = lmmodels.SkewedVoigtModel(prefix=f"{self.name}_")
            self._model.set_param_hint("gamma", value=0.5, expr=None)

        else:
            raise NotImplementedError(f"Unkown shape '{self._shape}'")

        self.params = self._model.make_params()
        if guess:
            self.params = self._model.guess(min=0, **kwargs)
        else:
            for k, v in kwargs.items():
                self.get_param(k).set(value=v, min=0, vary=True)

        # self.params += self._model.make_params()
        # self.get_param("fwhm").set(value=fwhm, min=0, vary=True)
        # self.get_param("amplitude").set(value=area, min=0, vary=True)
        # self.get_param("center").set(value=position, min=-np.inf, vary=True)

        """
        self.param_aliases = {"area": "amplitude", "fwhm": "fwhm", "position": "center"}
        if self._shape == "PseudoVoigt":
            self.param_aliases["alpha"] = "fraction"
            if alpha is None:
                alpha = 0.5
            self._model = models.PseudoVoigtModel(prefix=f"{self.name}_")
            self._model.set_param_hint("fraction", min=0, value=alpha)
        elif self._shape == "DoniachSunjic":
            self.param_aliases["alpha"] = "asym"
            if alpha is None:
                alpha = 0.1
            self._model = models.DoniachSunjicModel(prefix=f"{self.name}_")
            self._model.set_param_hint("asym", min=0, value=alpha)
        elif self._shape == "Voigt":
            self.param_aliases["alpha"] = "fwhm_l"
            if alpha is None:
                alpha = 0.5
            self._model = models.VoigtModel(prefix=f"{self.name}_")
            self._model.set_param_hint("fwhm_l", min=0, value=alpha)
        else:
            raise NotImplementedError(f"Unkown shape '{self._shape}'")

        self.params += self._model.make_params()
        self.get_param("fwhm").set(value=fwhm, min=0, vary=True)
        self.get_param("amplitude").set(value=area, min=0, vary=True)
        self.get_param("center").set(value=position, min=-np.inf, vary=True)
        """

    @property
    def shape(self):
        return self._shape

    # @shape.setter
    # def shape(self, value):
    #     if self._shape == value:
    #         return
    #     constraints = {}
    #     values = {}
    #     for param_alias in ("fwhm", "area", "position", "alpha"):
    #         constraints[param_alias] = self.get_constraints(param_alias)
    #         values[param_alias] = constraints[param_alias]["value"]
    #     # only change shape after getting constraints!
    #     if value in ("PseudoVoigt", "DoniachSunjic", "Voigt"):
    #         self._shape = value
    #     else:
    #         raise NotImplementedError
    #     self.initialize_model(**values)
    #     for param_alias in ("fwhm", "area", "position", "alpha"):
    #         self.set_constraints(param_alias, **constraints[param_alias])

    def get_param(self, param_alias, use_alias=False):
        """Shortcut for getting the Parameter object by the param alias."""
        if use_alias:
            aliases = {**self._default_aliases, **self.param_aliases}
            param_name = aliases.get(param_alias, param_alias)
        else:
            param_name = param_alias
        if param_name is None:
            raise ValueError(
                f"model '{self.shape}' does not support Parameter '{param_alias}'"
            )
        return self.params[f"{self.name}_{param_name}"]

    def clear_params(self):
        """Clear this peaks' parameters from the model."""
        pars_to_del = [
            par for par in self.params if re.match(rf"{self.name}_[a-z]+", par)
        ]
        for par in pars_to_del:
            self.params.pop(par)

    # def set_constraints(self, param_alias, **new):
    #     """Sets a constraint for param. Valid keys:
    #     value, vary, min, max, expr"""
    #     try:
    #         param = self.get_param(param_alias)
    #     except ValueError:
    #         return

    #     old = self.get_constraints(param_alias)
    #     # enforce min=0 for all parameters except position:
    #     if (
    #         old["min"] < 0
    #         and (new["min"] is None or new["min"] < 0)
    #         and param_alias != "position"
    #     ):
    #         new["min"] = 0
    #     # return if only None values are new
    #     if all(v is None for v in dict(set(new.items()) - set(old.items()))):
    #         return

    #     # if new["expr"]:
    #     #    new["expr"] = self.relation2expr(new["expr"], param_alias)

    #     try:
    #         param.set(**new)
    #         self.params.valuesdict()
    #     except (SyntaxError, NameError, TypeError):
    #         old["expr"] = ""
    #         param.set(**old)
    #         print(f"Invalid expression '{new['expr']}'")

    # def get_constraints(self, param_alias):
    #     """Returns a string containing min/max or expr."""
    #     try:
    #         param = self.get_param(param_alias)
    #     except ValueError:
    #         return self._default_constraints
    #     constraints = {
    #         "value": param.value,
    #         "min": param.min,
    #         "max": param.max,
    #         "vary": param.vary,
    #         "expr": self.expr2relation(param.expr),
    #     }
    #     return constraints

    # def expr2relation(self, expr):
    #     """Translates technical expr string into a human-readable relation."""
    #     if expr is None:
    #         return ""

    #     def param_repl(matchobj):
    #         """Replaces 'peakname_param' by 'peakname'"""
    #         param_key = matchobj.group(0)
    #         name = param_key.split("_")[0]
    #         if self in self.spectrum.peaks:
    #             return name
    #         return param_key

    #     regex = r"\b[A-Za-z][A-Za-z0-9]*_[a-z_]+"
    #     relation = re.sub(regex, param_repl, expr)
    #     return relation

    # def relation2expr(self, relation, param_alias):
    #     """Translates a human-readable arithmetic relation to an expr string."""

    #     def name_repl(matchobj):
    #         """Replaces 'peakname' by 'peakname_param' (searches case-insensitive)."""
    #         name = matchobj.group(0)
    #         name = name.upper()
    #         if name == self.name.upper():
    #             raise ValueError("Self-reference in peak constraint")
    #         for peak in self.spectrum.peaks:
    #             if peak.name.upper() == name:
    #                 other = peak
    #                 param_name = other.param_aliases[param_alias]
    #                 return f"{name}_{param_name}"
    #         return name

    #     regex = r"\b[A-Za-z][A-Za-z0-9]*"
    #     expr = re.sub(regex, name_repl, relation)
    #     return expr

    def get_measured_area(self):
        """Returns measured area under the peak."""
        return self._model.get_area(self.params)

    def get_measured_fwhm(self):
        """Returns measured fwhm of the peak."""
        return self._model.get_fwhm(self.params)
