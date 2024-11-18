"""Read CasaXPS ASCII data"""

import copy
from collections import OrderedDict
import matplotlib.pyplot as plt
import pandas as pd


class CasaXPS:
    "Export ASCII as Rows of Table"

    class Region:
        """_
        Class for CasaXPS energy regions e.g. transitions
        """

        def __init__(self, data):
            self.data = data

        def __getattr__(self, name):
            return self.data[name]

        def range(self, start, end):
            """_summary_

            Parameters
            ----------
            start : _type_
                _description_
            end : _type_
                _description_

            Returns
            -------
            _type_
                _description_
            """
            new_data = self.data.copy()

            new_data["be"] = self.be[(self.be > start) & (self.be < end)]
            new_data["cps"] = self.cps[(self.be > start) & (self.be < end)]
            new_data["bg"] = self.bg[(self.be > start) & (self.be < end)]
            new_data["envelope"] = self.envelope[(self.be > start) & (self.be < end)]

            new_comps = [comp.range(start, end) for comp in self.comps]
            new_data["comps"] = new_comps

            return type(self)(new_data)

        def info(self):
            print(self.name)
            data = []
            for comp in self.comps:
                data.append(
                    [comp.name, comp.position, comp.area, comp.fwhm, comp.lineshape]
                )

            print(
                pd.DataFrame(
                    data, columns=["Name", "Position", "Area", "FWHM", "Lineshape"]
                )
            )

        def rename(self, new_names):
            for name, comp in zip(new_names, self.comps):
                comp.name = name

        def plot(
            self,
            sub_bg=False,
            combine_label=True,
            ax=None,
            offset=0,
            env=True,
            comp_kwargs=None,
        ):
            """_summary_

            Parameters
            ----------
            sub_bg : bool, optional
                _description_, by default False
            combine_label : bool, optional
                _description_, by default True
            ax : _type_, optional
                _description_, by default None
            comp_kwargs : _type_, optional
                _description_, by default None

            Returns
            -------
            _type_
                _description_
            """

            ax = self._get_ax(ax)
            ax.set_xlabel("Binding energy in eV")
            ax.set_ylabel("Intensity in a.u.")
            ax.invert_xaxis()

            if sub_bg:
                _offset = -self.bg + offset
            else:
                _offset = offset

            comp_dict = OrderedDict()

            if combine_label:
                for index, comp in enumerate(self.comps):
                    if comp.name in comp_dict:
                        comp_dict[comp.name].comp_data["cps"] += comp.cps - self.bg
                    else:
                        comp_dict[comp.name] = copy.copy(comp)
            else:
                for index, comp in enumerate(self.comps):
                    comp_dict[index] = comp

            if comp_kwargs is None:
                comp_kwargs = [{}] * len(comp_dict)

            for index, (_, comp) in enumerate(comp_dict.items()):
                # alpha = comp_kwargs[index].pop("alpha", 0.75)
                label = comp_kwargs[index].pop("label", comp.name)

                # ax.fill_between(self.be,self.bg+offset,comp.cps+offset,interpolate=True,
                # alpha=alpha, label=label, **comp_kwargs[index])
                ax.plot(self.be, comp.cps + _offset, label=label, linewidth=3)

            ax.scatter(self.be, self.cps + _offset, facecolors="none", edgecolor="r")
            if env:
                ax.plot(self.be, self.envelope + _offset, "-", linewidth=2, color="k")
            ax.legend()

            return ax

        def _get_ax(self, ax, **kwargs):
            """Helper for preparing an axis object"""
            if ax is None:
                _, ax = plt.subplots(
                    figsize=kwargs.get("figsize", None), dpi=kwargs.get("dpi", 100)
                )
            if not kwargs.get("ticks", True):
                ax.set_xticks([])
                ax.set_yticks([])
                if kwargs.get("axis_off", False):
                    ax.set_axis_off()
            if kwargs.get("title", False):
                ax.set_title(kwargs["title"])
            if kwargs.get("xlabel", False):
                ax.set_xlabel(kwargs["xlabel"])
            if kwargs.get("ylabel", False):
                ax.set_ylabel(kwargs["ylabel"])
            return ax

    class Component:
        """_summary_"""

        def __init__(self, comp_data):
            """_summary_

            Parameters
            ----------
            comp_data : _type_
                _description_
            """
            self.comp_data = comp_data

        def __getattr__(self, key):
            return self.comp_data[key]

        @property
        def name(self):
            """_summary_

            Returns
            -------
            _type_
                _description_
            """
            return self.comp_data["name"]

        @name.setter
        def name(self, value):
            self.comp_data["name"] = value

        def range(self, start, end):
            """_summary_

            Parameters
            ----------
            start : _type_
                _description_
            end : _type_
                _description_

            Returns
            -------
            _type_
                _description_
            """
            new_be = self.be[(self.be > start) & (self.be < end)]
            new_cps = self.cps[(self.be > start) & (self.be < end)]
            new_comp_data = self.comp_data.copy()
            new_comp_data["be"] = new_be
            new_comp_data["cps"] = new_cps
            return type(self)(new_comp_data)

        def __copy__(self):
            return type(self)(self.comp_data.copy())

    def __init__(self, file):
        data = pd.read_csv(file, sep=",", skiprows=0, header=None).to_numpy()
        cycle_pos = []
        cycle_names = []

        for index, element in enumerate(data[5, :]):
            try:
                if "Cycle" in element:
                    if (
                        element not in cycle_names
                    ):  # Check if Cycle was already used to determine how many cycles there are
                        cycle_names.append(element)
                    cycle_pos.append(index)
            except:
                pass

        self.regions = []

        for cycle in cycle_pos:
            ii = cycle
            while "Envelope" not in data[5, ii]:
                ii = ii + 1
            cycle_end = ii
            cycle_name = data[5, cycle].split(":")[0:3]
            comps = []
            be = data[6:, cycle - 1].astype(float)
            cps = data[6:, cycle].astype(float)
            bg = data[6:, cycle_end - 1].astype(float)
            envelope = data[6:, cycle_end].astype(float)
            for comp in data.T[cycle + 1 : cycle_end - 1, :]:
                comps.append(
                    self.Component(
                        {
                            "name": comp[0],
                            "position": comp[1],
                            "fwhm": comp[2],
                            "area": comp[3],
                            "lineshape": comp[4],
                            "cps": comp[6:].astype(float),
                            "be": be,
                        }
                    )
                )
            # print(cycle_name)
            self.regions.append(
                self.Region(
                    {
                        "name": cycle_name,
                        "cps": cps,
                        "be": be,
                        "bg": bg,
                        "envelope": envelope,
                        "comps": comps,
                    }
                )
            )

    def info(self):
        for cycle in self.cycles:
            for region in cycle:
                region.info()


class CasaXPS_old:
    "Export ASCII as Rows of Table"

    class Region:
        """_
        Class for CasaXPS energy regions e.g. transitions
        """

        def __init__(self, data):
            self.data = data

        def __getattr__(self, name):
            return self.data[name]

        def range(self, start, end):
            """_summary_

            Parameters
            ----------
            start : _type_
                _description_
            end : _type_
                _description_

            Returns
            -------
            _type_
                _description_
            """
            new_data = self.data.copy()

            new_data["be"] = self.be[(self.be > start) & (self.be < end)]
            new_data["cps"] = self.cps[(self.be > start) & (self.be < end)]
            new_data["bg"] = self.bg[(self.be > start) & (self.be < end)]
            new_data["envelope"] = self.envelope[(self.be > start) & (self.be < end)]

            new_comps = [comp.range(start, end) for comp in self.comps]
            new_data["comps"] = new_comps

            return type(self)(new_data)

        def info(self):
            print(self.name)
            data = []
            for comp in self.comps:
                data.append(
                    [comp.name, comp.position, comp.area, comp.fwhm, comp.lineshape]
                )

            print(
                pd.DataFrame(
                    data, columns=["Name", "Position", "Area", "FWHM", "Lineshape"]
                )
            )

        def rename(self, new_names):
            for name, comp in zip(new_names, self.comps):
                comp.name = name

        def plot(self, sub_bg=False, combine_label=True, ax=None, comp_kwargs=None):
            """_summary_

            Parameters
            ----------
            sub_bg : bool, optional
                _description_, by default False
            combine_label : bool, optional
                _description_, by default True
            ax : _type_, optional
                _description_, by default None
            comp_kwargs : _type_, optional
                _description_, by default None

            Returns
            -------
            _type_
                _description_
            """

            ax = self._get_ax(ax)
            ax.set_xlabel("Binding energy in eV")
            ax.set_ylabel("Intensity in a.u.")
            ax.invert_xaxis()
            offset = 0

            if sub_bg:
                offset = -self.bg

            comp_dict = OrderedDict()

            if combine_label:
                for index, comp in enumerate(self.comps):
                    if comp.name in comp_dict:
                        comp_dict[comp.name].comp_data["cps"] += comp.cps - self.bg
                    else:
                        comp_dict[comp.name] = copy.copy(comp)
            else:
                for index, comp in enumerate(self.comps):
                    comp_dict[index] = comp

            if comp_kwargs is None:
                comp_kwargs = [{}] * len(comp_dict)

            for index, (_, comp) in enumerate(comp_dict.items()):
                # alpha = comp_kwargs[index].pop("alpha", 0.75)
                label = comp_kwargs[index].pop("label", comp.name)

                # ax.fill_between(self.be,self.bg+offset,comp.cps+offset,interpolate=True,
                # alpha=alpha, label=label, **comp_kwargs[index])
                ax.plot(self.be, comp.cps + offset, label=label, linewidth=3)

            ax.scatter(self.be, self.cps + offset, facecolors="none", edgecolor="r")
            ax.plot(self.be, self.envelope + offset, "-", linewidth=2, color="k")
            ax.legend()

            return ax

        def _get_ax(self, ax, **kwargs):
            """Helper for preparing an axis object"""
            if ax is None:
                _, ax = plt.subplots(
                    figsize=kwargs.get("figsize", None), dpi=kwargs.get("dpi", 100)
                )
            if not kwargs.get("ticks", True):
                ax.set_xticks([])
                ax.set_yticks([])
                if kwargs.get("axis_off", False):
                    ax.set_axis_off()
            if kwargs.get("title", False):
                ax.set_title(kwargs["title"])
            if kwargs.get("xlabel", False):
                ax.set_xlabel(kwargs["xlabel"])
            if kwargs.get("ylabel", False):
                ax.set_ylabel(kwargs["ylabel"])
            return ax

    class Component:
        """_summary_"""

        def __init__(self, comp_data):
            """_summary_

            Parameters
            ----------
            comp_data : _type_
                _description_
            """
            self.comp_data = comp_data

        def __getattr__(self, key):
            return self.comp_data[key]

        @property
        def name(self):
            """_summary_

            Returns
            -------
            _type_
                _description_
            """
            return self.comp_data["name"]

        @name.setter
        def name(self, value):
            self.comp_data["name"] = value

        def range(self, start, end):
            """_summary_

            Parameters
            ----------
            start : _type_
                _description_
            end : _type_
                _description_

            Returns
            -------
            _type_
                _description_
            """
            new_be = self.be[(self.be > start) & (self.be < end)]
            new_cps = self.cps[(self.be > start) & (self.be < end)]
            new_comp_data = self.comp_data.copy()
            new_comp_data["be"] = new_be
            new_comp_data["cps"] = new_cps
            return type(self)(new_comp_data)

        def __copy__(self):
            return type(self)(self.comp_data.copy())

    def __init__(self, file):
        data = pd.read_csv(file, sep=",", skiprows=0, header=None).to_numpy()
        self.cycles = []
        cycle_pos = []
        cycle_names = []

        for index, element in enumerate(data[5, :]):
            try:
                if "Cycle" in element:
                    if (
                        element not in cycle_names
                    ):  # Check if Cycle was already used to determine how many cycles there are
                        cycle_names.append(element)
                    cycle_pos.append(index)
            except:
                pass

        self.cycles = [[] for i in range(len(cycle_names))]

        for cycle in cycle_pos:
            ii = cycle
            while "Envelope" not in data[5, ii]:
                ii = ii + 1
            cycle_end = ii
            cycle_name = data[5, cycle].split(":")[0:3]
            comps = []
            be = data[6:, cycle - 1].astype(float)
            cps = data[6:, cycle].astype(float)
            bg = data[6:, cycle_end - 1].astype(float)
            envelope = data[6:, cycle_end].astype(float)
            for comp in data.T[cycle + 1 : cycle_end - 1, :]:
                comps.append(
                    self.Component(
                        {
                            "name": comp[0],
                            "position": comp[1],
                            "fwhm": comp[2],
                            "area": comp[3],
                            "lineshape": comp[4],
                            "cps": comp[6:].astype(float),
                            "be": be,
                        }
                    )
                )
            # print(cycle_name)
            self.cycles[int(cycle_name[0].split(" ")[1])].append(
                self.Region(
                    {
                        "name": cycle_name,
                        "cps": cps,
                        "be": be,
                        "bg": bg,
                        "envelope": envelope,
                        "comps": comps,
                    }
                )
            )

    def info(self):
        for cycle in self.cycles:
            for region in cycle:
                region.info()
