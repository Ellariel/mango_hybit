from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

import mosaik_api_v3
import pandas as pd
from loguru import logger
from mosaik_api_v3.types import (
    CreateResult,
    CreateResultChild,
    Meta,
    ModelDescription,
    OutputData,
    OutputRequest,
)

import pandapower as pp
import pandapower.networks


# For META, see below. (Non-conventional order do appease the type
# checker.)


class Simulator(mosaik_api_v3.Simulator):
    _sid: str
    """This simulator's ID."""
    _step_size: Optional[int]
    """The step size for this simulator. If ``None``, the simulator
    is running in event-based mode, instead.
    """
    _net: pp.pandapowerNet
    """The pandapowerNet for this simulator."""
    bus_auto_elements: pd.DataFrame
    """A dataframe listing the automatically generated loads and sgens
    to support connecting entities from other simulators directly to
    grid nodes.
    
    The index of this dataframe corresponds to the bus index. The two
    columns "load" and "sgen" contain the index of the corresponding
    load and sgen in the load and sgen element tables.
    """

    def __init__(self):
        super().__init__(META)
        self._net = None  # type: ignore  # set in init()
        self.bus_auto_elements = None  # type: ignore  # set in setup_done()

    def init(self, sid: str, time_resolution: float, step_size: Optional[int] = None):
        self._sid = sid
        if not step_size:
            self.meta["type"] = "event-based"
        self._step_size = step_size
        return self.meta

    def create(self, num: int, model: str, **model_params: Any) -> List[CreateResult]:
        if model == "Grid":
            if num != 1:
                raise ValueError("must create exactly one Grid entity")
            return [self.create_grid(**model_params)]

        if not self._net:
            raise ValueError(f"cannot create {model} entities before creating Grid")

        if model == "ControlledGen":
            return [self.create_controlled_gen(**model_params) for _ in range(num)]
        else:
            raise ValueError(f"no entities for the model {model} can be created")

    def create_grid(self, **params: Any) -> CreateResult:
        if self._net:
            raise ValueError("Grid was already created")

        self._net, self._profiles = load_grid(params)

        child_entities: List[CreateResult] = []
        for child_model, info in MODEL_TO_ELEMENT_INFO.items():
            for elem_tuple in self._net[info.elem].itertuples():
                child_entities.append(
                    {
                        "type": child_model,
                        "eid": f"{child_model}-{elem_tuple.Index}",
                        "rel": [
                            f"Bus-{getattr(elem_tuple, bus)}"
                            for bus in info.connected_buses
                        ],
                    }
                )

        return {
            "eid": "Grid",
            "type": "Grid",
            "children": child_entities,
            "rel": [],
        }

    def create_controlled_gen(self, bus: int) -> CreateResult:
        idx = pp.create_gen(self._net, bus, p_mw=0.0)
        return {
            "type": "ControlledGen",
            "eid": f"ControlledGen-{idx}",
            "children": [],
            "rel": [f"Bus-{bus}"],
        }

    def setup_done(self):
        # Create "secret" loads and sgens that are used when the user
        # provides real and reactive power directly to grid nodes.
        load_indices = pp.create_loads(self._net, self._net.bus.index, 0.0)
        sgen_indices = pp.create_sgens(self._net, self._net.bus.index, 0.0)
        self.bus_auto_elements = pd.DataFrame(
            {
                "load": load_indices,
                "sgen": sgen_indices,
            },
            index=self._net.bus.index,
        )

    def get_model_and_idx(self, eid: str) -> Tuple[str, int]:
        # TODO: Maybe add a benchmark whether caching this in a dict is
        # faster
        model, idx_str = eid.split("-")
        return (model, int(idx_str))

    def step(self, time, inputs, max_advance):
        if self._profiles:
            # TODO: Division by 900 here assumes a time_resolution of 1.
            apply_profiles(self._net, self._profiles, time // 900)
        for eid, data in inputs.items():
            model, idx = self.get_model_and_idx(eid)
            info = MODEL_TO_ELEMENT_INFO[model]
            for attr, values in data.items():
                attr_info = info.in_attrs[attr]
                self._net[attr_info.target_elem or info.elem].at[
                    attr_info.idx_fn(idx, self), attr_info.column
                ] = attr_info.aggregator(values.values())

        pp.runpp(self._net)
        if self._step_size:
            return time + self._step_size

    def get_data(self, outputs: OutputRequest) -> OutputData:
        return {eid: self.get_entity_data(eid, attrs) for eid, attrs in outputs.items()}

    def get_entity_data(self, eid: str, attrs: List[str]) -> Dict[str, Any]:
        model, idx = self.get_model_and_idx(eid)
        info = MODEL_TO_ELEMENT_INFO[model]
        elem_table = self._net[f"res_{info.elem}"]
        return {
            attr: elem_table.at[idx, info.out_attr_to_column[attr]] for attr in attrs
        }


@dataclass
class InAttrInfo:
    """Specificaction of an input attribute of a model."""

    column: str
    """The name of the column in the target element's dataframe
    corresponding to this attribute.
    """
    target_elem: Optional[str] = None
    """The name of the pandapower element to which this attribute's
    inputs are written. (This might not be the element type
    corresponding to the model to support connecting loads and sgens
    directly to the buses.)
    If ``None``, use the element corresponding to the model.
    """
    idx_fn: Callable[[int, Simulator], int] = lambda idx, sim: idx
    """A function to transform the entity ID's index part into the
    index for the target_df.
    """
    aggregator: Callable[[Iterable[Any]], Any] = sum
    """The function that is used for aggregation if several values are
    given for this attribute.
    """


@dataclass
class ModelElementInfo:
    """Specification of the pandapower element that is represented by
    a (mosaik) model of this simulator.
    """

    elem: str
    """The name of the pandapower element corresponding to this model.
    """
    connected_buses: List[str]
    """The names of the columns specifying the buses to which this
    element is connected.
    """
    in_attrs: Dict[str, InAttrInfo]
    """Mapping each input attr to the corresponding column in the
    element's dataframe and an aggregation function.
    """
    out_attr_to_column: Dict[str, str]
    """Mapping each output attr to the corresponding column in the
    element's result dataframe.
    """
    createable: bool = False
    """Whether this element can be created by the user."""
    params: List[str] = field(default_factory=list)
    """The mosaik params that may be given when creating this element.
    (Only sensible if ``createable=True``.)
    """


MODEL_TO_ELEMENT_INFO = {
    "Bus": ModelElementInfo(
        elem="bus",
        connected_buses=[],
        in_attrs={
            "P_gen[MW]": InAttrInfo(
                column="p_mw",
                target_elem="sgen",
                idx_fn=lambda idx, sim: sim.bus_auto_elements.at[idx, "sgen"],
            ),
            "P_load[MW]": InAttrInfo(
                column="p_mw",
                target_elem="load",
                idx_fn=lambda idx, sim: sim.bus_auto_elements.at[idx, "load"],
            ),
            "Q_gen[MVar]": InAttrInfo(
                column="q_mvar",
                target_elem="sgen",
                idx_fn=lambda idx, sim: sim.bus_auto_elements.at[idx, "sgen"],
            ),
            "Q_load[MVar]": InAttrInfo(
                column="q_mvar",
                target_elem="load",
                idx_fn=lambda idx, sim: sim.bus_auto_elements.at[idx, "load"],
            ),
        },
        out_attr_to_column={
            "P[MW]": "p_mw",
            "Q[MVar]": "q_mvar",
            "Vm[pu]": "vm_pu",
            "Va[deg]": "va_degree",
        },
    ),
    "Load": ModelElementInfo(
        elem="load",
        connected_buses=["bus"],
        in_attrs={},
        out_attr_to_column={
            "P[MW]": "p_mw",
            "Q[MVar]": "q_mvar",
        },
    ),
    "StaticGen": ModelElementInfo(
        elem="sgen",
        connected_buses=["bus"],
        in_attrs={},
        out_attr_to_column={
            "P[MW]": "p_mw",
            "Q[MVar]": "q_mvar",
        },
    ),
    "Gen": ModelElementInfo(
        elem="gen",
        connected_buses=["bus"],
        in_attrs={},
        out_attr_to_column={
            "P[MW]": "p_mw",
            "Q[MVar]": "q_mvar",
            "Va[deg]": "va_degree",
            "Vm[pu]": "vm_pu",
        },
    ),
    "ExternalGrid": ModelElementInfo(
        elem="ext_grid",
        connected_buses=["bus"],
        in_attrs={},
        out_attr_to_column={
            "P[MW]": "p_mw",
            "Q[MVar]": "q_mvar",
        }
    ),
    "ControlledGen": ModelElementInfo(
        elem="gen",
        connected_buses=["bus"],
        in_attrs={
            "P[MW]": InAttrInfo(
                column="p_mw",
            )
        },
        out_attr_to_column={},
        createable=True,
        params=["bus"],
    ),
    "Line": ModelElementInfo(
        elem="line",
        connected_buses=["from_bus", "to_bus"],
        in_attrs={},
        out_attr_to_column={
            "I[kA]": "i_ka",
            "loading[%]": "loading_percent",
        },
    ),
}


# Generate mosaik model descriptions out of the MODEL_TO_ELEMENT_INFO
ELEM_META_MODELS: Dict[str, ModelDescription] = {
    model: {
        "public": info.createable,
        "params": info.params,
        "attrs": list(info.in_attrs.keys()) + list(info.out_attr_to_column.keys()),
        "any_inputs": False,
        "persistent": [],
        "trigger": [],
    }
    for model, info in MODEL_TO_ELEMENT_INFO.items()
}


META: Meta = {
    "api_version": "3.0",
    "type": "time-based",
    "models": {
        "Grid": {
            "public": True,
            "params": ["json", "xlsx", "net", "simbench", "network_function", "params"],
            "attrs": [],
            "any_inputs": False,
            "persistent": [],
            "trigger": [],
        },
        **ELEM_META_MODELS,
    },
    "extra_methods": [],
}


def apply_profiles(net: pp.pandapowerNet, profiles: Any, step: int):
    """Apply element profiles for the given step to the grid.

    :param profiles: profiles for elements in the format returned by
        simbench's ``get_absolute_values`` function.
    :param step: the time step to apply
    """
    for (elm, param), series in profiles.items():
        net[elm].loc[:, param].update(series.loc[step])  # type: ignore


def load_grid(params: Dict[str, Any]) -> Tuple[pp.pandapowerNet, Any]:
    """Load a grid and the associated element profiles (if any).

    :param params: A dictionary describing which grid to load. It should
        contain one of the following keys (or key combinations).

        - `"net"` where the corresponding value is a pandapowerNet
        - `"json"` where the value is the name of a JSON file in
          pandapower JSON format
        - `"xlsx"` where the value is the name of an Excel file
        - `"network_function"` giving the name of a function in
          pandapower.networks. In this case, the additional key
          `"params"` may be given to specify the kwargs to that function 
        - `"simbench"` giving a simbench ID (if simbench is installed)

    :return: a tuple consisting of a :class:`pandapowerNet` and "element
        profiles" in the form that is returned by simbench's
        get_absolute_values function (or ``None`` if the loaded grid
        is not a simbench grid).

    :raises ValueError: if multiple keys are given in `params`
    """
    found_sources: Set[str] = set()
    result: Optional[Tuple[pp.pandapowerNet, Any]] = None

    # Accept a pandapower grid
    if net := params.get("net", None):
        if isinstance(net, pp.pandapowerNet):
            result = (net, None)
            found_sources.add("net")
        else:
            raise ValueError("net is not a pandapowerNet instance")

    if json_path := params.get("json", None):
        result = (pp.from_json(json_path), None)
        found_sources.add("json")

    if xlsx_path := params.get("xlsx", None):
        result = (pp.from_excel(xlsx_path), None)
        found_sources.add("xlsx")

    if network_function := params.get("network_function", None):
        result = (
            getattr(pandapower.networks, network_function)(**params.get("params", {})),
            None,
        )
        found_sources.add("network_function")

    if simbench_id := params.get("simbench", None):
        import simbench as sb

        net = sb.get_simbench_net(simbench_id)
        profiles = sb.get_absolute_values(net, profiles_instead_of_study_cases=True)
        profiles = {
            (elm, col): df
            for (elm, col), df in profiles.items()
            if not net[elm].empty
        }
        result = (net, profiles)
        found_sources.add("simbench")

    if len(found_sources) != 1 or not result:
        raise ValueError(
            f"too many or too few sources specified for grid, namely: {found_sources}"
        )

    return result