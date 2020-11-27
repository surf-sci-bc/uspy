"""Spectrum class represents spectrum data."""
# pylint: disable=too-many-instance-attributes
# pylint: disable=missing-docstring
# pylint: disable=invalid-name

import re

import numpy as np
from lmfit import Parameters
from lmfit.models import ConstantModel#, PseudoVoigtModel

from agfalta.xps import processing, models, io


class XPSSpectrum:
    """
    Holds data from one single spectrum.
    """
    background_types = (None, "linear", "shirley", "tougaard")
    norm_types = (None, "highest", "high_energy", "low_energy", "manual", "energy")
    _defaults = {
        "energy_scale": "binding",
        "photon_energy": 0,
        "notes": "",
        "sweeps": np.nan,
        "dwelltime": np.nan,
        "pass_energy": np.nan,
    }
    def __init__(self, path, idx=0):
        self.path = path
        self._meta = {}
        self._energy = None
        self._intensity = None

        self.parse_txt(idx)

        if self._meta["energy_scale"] == "kinetic":
            print("energy scale is not binding energy, trying to convert...")
            self._energy = self._meta["photon_energy"] - self._energy
        if len(self._energy) != len(self._intensity) or self._energy.ndim != 1:
            raise ValueError("energy and intensity array sizes differ")
        self._energy, self._intensity = processing.make_increasing(self._energy, self._intensity)
        self._energy, self._intensity = processing.make_equidistant(self._energy, self._intensity)

        self._background = np.zeros_like(self._energy)
        self._background_type = None
        self._background_bounds = np.array([])

        self.energy_calibration = 0

        self._norm_divisor = 1.0
        self._norm_type = None

    def parse_txt(self, idx):
        try:
            specdict = io.parse_spectrum_file(self.path)[idx]
        except FileNotFoundError:
            raise ValueError(f"{self.path} does not exist or has wrong file format")
        try:
            self._energy = specdict.pop("energy")
            self._intensity = specdict.pop("intensity")
        except KeyError:
            raise ValueError(f"{self.path} cannot be parsed correctly")
        self._meta = self._defaults.copy()
        self._meta.update(specdict)

    def __getattr__(self, attr):
        if attr in ("path", "_meta", "_energy", "_intensity"):
            raise AttributeError
        try:
            return self._meta[attr]
        except KeyError:
            raise AttributeError(f"No attribute named {attr}")

    def __setattr__(self, attr, value):
        if hasattr(self, "_meta") and attr in self._meta:
            self._meta[attr] = value
        else:
            super().__setattr__(attr, value)

    # Energy-related
    @property
    def energy(self):
        return self._energy + self.energy_calibration

    @property
    def kinetic_energy(self):
        return self.photon_energy - self._energy + self.energy_calibration

    # Intensity-related
    @property
    def intensity(self):
        return self._intensity / self._norm_divisor

    def intensity_at_E(self, energy):
        return processing.intensity_at_energy(self.energy, self.intensity, energy)

    def norm(self, norm_type="manual", value=1.0):
        if norm_type not in self.norm_types:
            raise ValueError(f"Invalid normalization type '{norm_type}'")
        if norm_type is None:
            self._norm_divisor = 1.0
        elif norm_type == "highest":
            self._norm_divisor = self._intensity.max()
        elif norm_type == "manual":
            self._norm_divisor = value
        elif norm_type == "energy":
            self._norm_divisor = 1.0
            self._norm_divisor = self.intensity_at_E(value)
        else:
            span = self._intensity.max() - self._intensity.min()
            if norm_type == "low_energy":
                idx = np.argmax(np.abs(self._intensity - self._intensity[0]) > span * 0.05)
                self._norm_divisor = self._intensity[:idx:].mean()
            elif norm_type == "high_energy":
                idx = np.argmax(np.abs(self._intensity[::-1] - self._intensity[-1]) > span * 0.05)
                self._norm_divisor = self._intensity[-1:idx:-1].mean()
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
    @background_type.setter
    def background_type(self, value):
        if value not in self.background_types:
            raise ValueError(f"Background type {value} not valid")
        self._background = processing.calculate_background(
            value, self._background_bounds, self._energy, self._intensity
        )
        self._background_type = value

    @property
    def background_bounds(self):
        return self._background_bounds + self._energy_calibration
    @background_bounds.setter
    def background_bounds(self, value):
        """Only even-length numeral sequence-types are valid."""
        if len(value) % 2 != 0:
            raise ValueError("Background bounds must be pairwise.")
        self._background_bounds = np.sort(np.array(value)) - self._energy_calibration
        self._background_bounds = self._background_bounds.clip(
            self._energy.min(), self._energy.max())
        self._background = processing.calculate_background(
            self.background_type, self.background_bounds, self.energy, self._intensity)


class ModeledSpectrum(XPSSpectrum):
    """Holds information on the Fit and provides methods for fitting."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.params = Parameters()
        self._peaks = []

    @property
    def peaks(self):
        """Returns peaks."""
        return self._peaks.copy()

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
        model = ConstantModel(prefix="BASE_")
        model.set_param_hint("c", vary=False, value=0)
        self.params += model.make_params()

        for peak in self._peaks:
            model += peak.model
        return model

    def do_fit(self):
        """Returns the fitted cps values."""
        with processing.IgnoreUnderflow():
            result = self.model.fit(
                self.intensity - self.background,
                self.params,
                x=self.energy
            )
        self.params.update(result.params)

    def add_peak(self, name, **kwargs):
        """
        Add a peak with given parameters. Valid parameters:
        area, fwhm, position and model specific parameters:
            PseudoVoigt: fraction, gausswidth
        """
        if name in [peak.name for peak in self._peaks]:
            raise ValueError("Peak already exists")
        if "fwhm" not in kwargs and "height" in kwargs and "angle" in kwargs:
            kwargs["fwhm"] = models.pah2fwhm(
                kwargs["position"],
                kwargs["angle"],
                kwargs["height"],
                kwargs["shape"]
            )
        if "area" not in kwargs and "height" in kwargs:
            kwargs["area"] = models.pah2area(
                kwargs["position"],
                kwargs["angle"],
                kwargs["height"],
                kwargs["shape"]
            )
        kwargs.pop("angle", None)
        kwargs.pop("height", None)
        peak = Peak(name, self, **kwargs)
        self._peaks.append(peak)
        return peak

    def remove_peak(self, peak):
        """Remove a peak specified by its name."""
        peak.clear_params()
        self._peaks.remove(peak)


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
    _signals = ("changed-peak", "changed-peak-meta")
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
        "expr": ""
    }
    shapes = ["PseudoVoigt", "DoniachSunjic", "Voigt"]

    def __init__(
            self, name, spectrum,
            area=None, fwhm=None, position=None, alpha=None,
            shape="PseudoVoigt",
        ):
        # pylint: disable=too-many-arguments
        super().__init__()
        self._name = name
        self.spectrum = spectrum
        self.params = spectrum.params
        self._shape = shape
        self.label = f"Peak {name}"
        if None in (area, fwhm, position):
            raise ValueError("Required attribute(s) missing")

        self._model = None
        self.initialize_model(area, fwhm, position, alpha)

    @property
    def name(self):
        return self._name

    @property
    def model(self):
        return self._model

    @property
    def intensity(self):
        with processing.IgnoreUnderflow():
            intensity = self._model.eval(
                params=self.params,
                x=self.spectrum.energy
            )
        return intensity

    def intensity_at_E(self, energy):
        with processing.IgnoreUnderflow():
            intensity = self._model.eval(params=self.params, x=energy)
        return intensity

    def initialize_model(self, area, fwhm, position, alpha):
        self.clear_params()
        self.param_aliases = {
            "area": "amplitude",
            "fwhm": "fwhm",
            "position": "center"
        }
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

    @property
    def shape(self):
        return self._shape
    @shape.setter
    def shape(self, value):
        if self._shape == value:
            return
        constraints = {}
        values = {}
        for param_alias in ("fwhm", "area", "position", "alpha"):
            constraints[param_alias] = self.get_constraints(param_alias)
            values[param_alias] = constraints[param_alias]["value"]
        # only change shape after getting constraints!
        if value in ("PseudoVoigt", "DoniachSunjic", "Voigt"):
            self._shape = value
        else:
            raise NotImplementedError
        self.initialize_model(**values)
        for param_alias in ("fwhm", "area", "position", "alpha"):
            self.set_constraints(param_alias, **constraints[param_alias])

    def get_param(self, param_alias, use_alias=True):
        """Shortcut for getting the Parameter object by the param alias."""
        if use_alias:
            aliases = {**self._default_aliases, **self.param_aliases}
            param_name = aliases.get(param_alias, param_alias)
        if param_name is None:
            raise ValueError(f"model '{self.shape}' does not support Parameter '{param_alias}'")
        return self.params[f"{self.name}_{param_name}"]

    def clear_params(self):
        """Clear this peaks' parameters from the model."""
        pars_to_del = [
            par for par in self.params
            if re.match(fr"{self.name}_[a-z]+", par)
        ]
        for par in pars_to_del:
            self.params.pop(par)

    def set_constraints(self, param_alias, **new):
        """Sets a constraint for param. Valid keys:
        value, vary, min, max, expr"""
        try:
            param = self.get_param(param_alias)
        except ValueError:
            return

        old = self.get_constraints(param_alias)
        # enforce min=0 for all parameters except position:
        if (old["min"] < 0
                and (new["min"] is None or new["min"] < 0)
                and param_alias != "position"):
            new["min"] = 0
        # return if only None values are new
        if all(v is None for v in dict(set(new.items()) - set(old.items()))):
            return

        if new["expr"]:
            new["expr"] = self.relation2expr(new["expr"], param_alias)

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
            "expr": self.expr2relation(param.expr)
        }
        return constraints

    def expr2relation(self, expr):
        """Translates technical expr string into a human-readable relation."""
        if expr is None:
            return ""
        def param_repl(matchobj):
            """Replaces 'peakname_param' by 'peakname'"""
            param_key = matchobj.group(0)
            name = param_key.split("_")[0]
            if self in self.spectrum.peaks:
                return name
            return param_key
        regex = r"\b[A-Za-z][A-Za-z0-9]*_[a-z_]+"
        relation = re.sub(regex, param_repl, expr)
        return relation

    def relation2expr(self, relation, param_alias):
        """Translates a human-readable arithmetic relation to an expr string."""
        def name_repl(matchobj):
            """Replaces 'peakname' by 'peakname_param' (searches case-insensitive)."""
            name = matchobj.group(0)
            name = name.upper()
            if name == self.name.upper():
                raise ValueError("Self-reference in peak constraint")
            for peak in self.spectrum.peaks:
                if peak.name.upper() == name:
                    other = peak
                    param_name = other.param_aliases[param_alias]
                    return f"{name}_{param_name}"
            return name
        regex = r"\b[A-Za-z][A-Za-z0-9]*"
        expr = re.sub(regex, name_repl, relation)
        return expr

    def get_measured_area(self):
        """Returns measured area under the peak."""
        return self._model.get_area(self.params)

    def get_measured_fwhm(self):
        """Returns measured fwhm of the peak."""
        return self._model.get_fwhm(self.params)
