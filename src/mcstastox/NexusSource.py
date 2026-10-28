# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Mccode-dev contributors (https://github.com/mccode-dev)
"""Source specifications and row-preserving projections for NeXus writing.

Event iteration delegates to the existing McStas reader. Sampling is deliberately
not executed here: a writer must call ``sample_event_chunks`` once per shared
sampled EventSource, then apply its columns to aligned output batches. The reader
loads a row block's full table width, and the sampler retains its complete sample.
Neither interface promises a fixed byte budget or read-batch-independent seeds.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field, replace
from fractions import Fraction
from itertools import islice
from numbers import Number
from os import PathLike
from pathlib import Path
from types import MappingProxyType
from typing import Any

import h5py
import numpy as np
from numpy.typing import DTypeLike

from .LoadFile import Data
from .ReadNeXus import McStasNeXus, _validate_chunk_size
from .Sampling import SamplingSettings

__all__ = [
    "ChunkOperation",
    "ColumnOperation",
    "EventColumn",
    "EventSource",
    "EventTransform",
    "SourceContext",
    "event_source",
]


class SourceContext:
    """Adapt a full file path, borrowed Read/Data/context, or no source.

    Paths open an owned ``Data(parent, filename=basename)`` immediately, using
    entry 1 unless overridden. Borrowed handles retain their selected entry and
    are never closed here. ``None`` is valid for static-only writing and fails
    only on source access. Inputs must remain unchanged throughout execution;
    the reader's SWMR mode is not a snapshot.
    """

    def __init__(
        self,
        source: Data | SourceContext | str | PathLike[str] | None,
        *,
        entry_number: int | None = None,
    ) -> None:
        if entry_number is not None:
            if isinstance(entry_number, bool) or not isinstance(entry_number, int):
                raise TypeError("entry_number must be an integer or None")
            if entry_number < 1:
                raise ValueError("entry_number must be positive")
        self._owned = False
        self._data: Data | None
        if isinstance(source, SourceContext):
            self._data = source._data
        elif isinstance(source, Data) or source is None:
            self._data = source
        elif isinstance(source, (str, PathLike)):
            path = Path(source)
            self._data = Data(
                path.parent,
                filename=path.name,
                entry_number=1 if entry_number is None else entry_number,
            )
            self._owned = True
        else:
            raise TypeError(
                "source must be a Read/Data, SourceContext, full path, or None"
            )
        if self._data is not None:
            self._require_data()
            if not self._owned and entry_number is not None:
                if entry_number != self.mcstas.entry_number:
                    raise ValueError(
                        f"Borrowed source selects {self.entry_name}; "
                        f"entry_number={entry_number} conflicts with that entry"
                    )

    def _require_data(self) -> Data:
        if self._data is None:
            raise ValueError("This operation requires a source; source is None")
        if not self._data.file.id.valid:
            raise ValueError("The source file is closed")
        return self._data

    @property
    def raw(self) -> h5py.File:
        """The file-wide HDF5 handle, not redirected to the selected entry."""
        return self._require_data().file

    @property
    def mcstas(self) -> McStasNeXus:
        return self._require_data().file_object

    @property
    def entry_name(self) -> str:
        return self.mcstas.entry_name

    @property
    def entry(self) -> _EntryAccessor:
        return _EntryAccessor(self)

    def _dataset(self, path: Any) -> h5py.Dataset:
        if not isinstance(path, str) or not path.startswith("/"):
            raise ValueError("File-wide source paths must be absolute HDF5 paths")
        dataset = self.raw[path]
        if not isinstance(dataset, h5py.Dataset):
            raise TypeError(f"Source path {path!r} is not a dataset")
        return dataset

    def read(self, path: str, selection: Any = ()) -> Any:
        """Read without coercion, with h5py selection and selected-shape semantics.

        ``()`` reads the whole dataset; integers remove axes, slices retain axes,
        and tuples select multiple axes. Scalar and length-one data stay distinct.
        """
        return self._dataset(path)[selection]

    def read_text(self, path: str, selection: Any = ()) -> Any:
        """Decode a string dataset as UTF-8, never parsing numeric-looking text."""
        dataset = self._dataset(path)
        if h5py.check_string_dtype(dataset.dtype) is None:
            raise TypeError(f"Source path {path!r} is not a text dataset")
        return dataset.asstr(encoding="utf-8")[selection]

    def iter_dataset(
        self, path: str, *, read_chunk_size: int = 100_000, selection: Any = ()
    ) -> Iterator[Any]:
        """Yield raw batches along the first retained axis of a basic selection.

        The batch size counts rows, not bytes or HDF5 storage chunks. Integers,
        positive-step slices, and one ellipsis are supported. Scalar selections
        yield once; empty first axes yield nothing. Trailing axes are not tiled.
        """
        _validate_chunk_size(read_chunk_size, allow_none=False)
        dataset = self._dataset(path)
        if dataset.shape is None:
            raise ValueError(f"Cannot iterate null dataset {path!r}")
        indices = selection if isinstance(selection, tuple) else (selection,)
        ellipses = sum(index is Ellipsis for index in indices)
        missing = dataset.ndim - len(indices) + ellipses
        if ellipses > 1 or missing < 0:
            raise IndexError("Invalid dataset selection rank")
        expanded: list[Any] = []
        for index in indices:
            expanded.extend([slice(None)] * missing if index is Ellipsis else [index])
        if not ellipses:
            expanded.extend([slice(None)] * missing)
        axis = None
        for dimension, (index, size) in enumerate(
            zip(expanded, dataset.shape, strict=True)
        ):
            if isinstance(index, (int, np.integer)) and not isinstance(
                index, (bool, np.bool_)
            ):
                if not -size <= index < size:
                    raise IndexError("Dataset selection index is out of range")
                expanded[dimension] = int(index) % size
            elif isinstance(index, slice):
                start, stop, step = index.indices(size)
                if step < 1:
                    raise ValueError("Dataset iteration requires positive slice steps")
                expanded[dimension] = slice(start, stop, step)
                if axis is None:
                    axis = dimension
            else:
                raise TypeError("Dataset iteration supports only integers and slices")
        if axis is None:
            yield dataset[tuple(expanded)]
            return
        rows = expanded[axis]
        for offset in range(
            0, len(range(rows.start, rows.stop, rows.step)), read_chunk_size
        ):
            self._require_data()
            expanded[axis] = slice(
                rows.start + offset * rows.step,
                min(rows.stop, rows.start + (offset + read_chunk_size) * rows.step),
                rows.step,
            )
            yield dataset[tuple(expanded)]

    def close(self) -> None:
        if self._owned and self._data is not None:
            self._data.close()

    def __enter__(self) -> SourceContext:  # noqa: PYI034 (Python 3.10)
        if self._data is not None:
            self._require_data()
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()


@dataclass(frozen=True)
class _EntryAccessor:
    _context: SourceContext

    @property
    def raw(self) -> h5py.Group:
        return self._context.raw[self._context.entry_name]

    def _path(self, path: Any) -> str:
        if not isinstance(path, str) or not path or path.startswith("/"):
            raise ValueError("Entry source paths must be nonempty relative HDF5 paths")
        if ".." in path.split("/"):
            raise ValueError("Entry source paths cannot contain '..'")
        return f"/{self._context.entry_name}/{path}"

    def read(self, path: str, selection: Any = ()) -> Any:
        return self._context.read(self._path(path), selection=selection)

    def read_text(self, path: str, selection: Any = ()) -> Any:
        return self._context.read_text(self._path(path), selection=selection)

    def iter_dataset(
        self, path: str, *, read_chunk_size: int = 100_000, selection: Any = ()
    ) -> Iterator[Any]:
        return self._context.iter_dataset(
            self._path(path), read_chunk_size=read_chunk_size, selection=selection
        )


def _names(value: Any, name: str) -> tuple[str, ...]:
    if isinstance(value, str):
        value = (value,)
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{name} must be a string, list, or tuple of strings")
    if not value:
        raise ValueError(f"{name} must not be empty")
    if any(
        not isinstance(item, str) or not item or any(c.isspace() for c in item)
        for item in value
    ):
        raise ValueError(f"{name} must contain nonempty single-token strings")
    if len(set(value)) != len(value):
        raise ValueError(f"{name} must not contain duplicates")
    return tuple(value)


def _numeric_dtype(dtype: DTypeLike) -> np.dtype:
    result = np.dtype(dtype)
    if result.kind not in "iufc":
        raise TypeError(f"Expected a numeric dtype, got {result}")
    return result


def _event_arrays(chunk: Mapping[str, Any]) -> dict[str, np.ndarray]:
    arrays = {key: np.asarray(value) for key, value in chunk.items()}
    if any(value.ndim != 1 for value in arrays.values()):
        raise ValueError("Event arrays must be one-dimensional")
    if len({len(value) for value in arrays.values()}) > 1:
        raise ValueError("Event arrays must have the same row count")
    return arrays


@dataclass(frozen=True, repr=False)
class ChunkOperation:
    """A custom dictionary callback with an immutable, complete output schema.

    Schema inspection never executes the callback. Row correspondence, retained
    weights, determinism, and batch independence remain caller responsibilities.
    """

    callback: Callable[[dict[str, np.ndarray]], Mapping[str, Any]]
    schema: Mapping[str, DTypeLike]
    inputs: tuple[str, ...] | None = None
    label: str | None = None

    def __post_init__(self) -> None:
        if not callable(self.callback):
            raise TypeError("callback must be callable")
        if not isinstance(self.schema, Mapping):
            raise TypeError(
                "schema must be a mapping of variable names to numeric dtypes"
            )
        _names(tuple(self.schema), "schema variables")
        object.__setattr__(
            self,
            "schema",
            MappingProxyType(
                {key: _numeric_dtype(value) for key, value in self.schema.items()}
            ),
        )
        if self.inputs is not None:
            object.__setattr__(self, "inputs", _names(self.inputs, "inputs"))
        if self.label is not None and not isinstance(self.label, str):
            raise TypeError("description must be a string or None")

    def apply(
        self,
        chunk: Mapping[str, np.ndarray],
        *,
        allow_filter: bool = False,
        context: str = "Chunk transform",
    ) -> dict[str, np.ndarray]:
        try:
            names = tuple(chunk) if self.inputs is None else self.inputs
            private = {}
            for name in names:
                private[name] = chunk[name].copy()
                private[name].flags.writeable = False
            result = self.callback(private)
            if not isinstance(result, Mapping):
                raise TypeError(
                    "Chunk callback must return a dictionary matching its schema"
                )
            if set(result) != set(self.schema):
                raise ValueError(
                    "Chunk callback output keys must exactly match its schema"
                )
            arrays = _event_arrays(result)
            count = len(next(iter(chunk.values())))
            actual = len(next(iter(arrays.values())))
            if (allow_filter and actual > count) or (
                not allow_filter and actual != count
            ):
                rule = "not exceed" if allow_filter else "preserve"
                raise ValueError(f"Chunk callback must {rule} the input row count")
            return {
                name: _checked_cast(arrays[name], dtype)
                for name, dtype in self.schema.items()
            }
        except (ValueError, TypeError, KeyError, OverflowError, RuntimeError) as error:
            raise type(error)(f"{context} ({self.description}): {error}") from error

    @property
    def description(self) -> str:
        name = getattr(self.callback, "__name__", "")
        text = "custom callback" + (
            f" {name[:60]}" if name and name != "<lambda>" else ""
        )
        if self.label is not None:
            text += f" (user description: {self.label[:100]})"
        schema = ", ".join(
            f"{key[:40]}: {dtype}" for key, dtype in islice(self.schema.items(), 5)
        )
        if len(self.schema) > 5:
            schema += f", ... ({len(self.schema) - 5} more)"
        return f"{text}; declared output schema={{{schema}}}"

    def __repr__(self) -> str:
        return f"ChunkOperation({self.description})"


@dataclass(frozen=True, eq=False, repr=False)
class EventSource:
    """An immutable, identity-based shared event execution specification.

    ``None`` components retain the reader's default: components with pixel IDs,
    in reader order. Sampling does not run in ``iter_chunks``. Its original
    weights are included as hidden dependencies, with no implicit zero filtering.
    """

    component_name: str | tuple[str, ...] | None = None
    variables: tuple[str, ...] = ("p", "t", "id")
    read_chunk_size: int = 100_000
    filter_zeros: bool = False
    weight_key: str = "p"
    sampling: SamplingSettings | None = None
    preprocessors: tuple[ChunkOperation, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "variables", _names(self.variables, "variables"))
        if self.component_name is not None:
            names = _names(self.component_name, "component_name")
            if not isinstance(self.component_name, str):
                object.__setattr__(self, "component_name", names)
        _validate_chunk_size(self.read_chunk_size, allow_none=False)
        if not isinstance(self.filter_zeros, bool):
            raise TypeError("filter_zeros must be a bool")
        _names(self.weight_key, "weight_key")
        if not isinstance(self.weight_key, str):
            raise TypeError("weight_key must be a string")
        if self.sampling is not None and not isinstance(
            self.sampling, SamplingSettings
        ):
            raise TypeError("sampling must be a SamplingSettings instance")
        object.__setattr__(self, "preprocessors", tuple(self.preprocessors))
        for stage in self.preprocessors:
            if not isinstance(stage, ChunkOperation) or stage.inputs is not None:
                raise TypeError("preprocessors must be complete-schema ChunkOperations")
            if self.sampling is not None:
                weight = self.sampling.weight_key
                if weight not in stage.schema:
                    raise ValueError(
                        f"Preprocessing schema must retain sampling weight {weight!r}"
                    )
                if stage.schema[weight].kind not in "iuf":
                    raise TypeError("Preprocessed sampling weights must be real")

    @property
    def required_variables(self) -> tuple[str, ...]:
        variables = list(self.variables)
        if self.sampling is not None and self.sampling.weight_key not in variables:
            variables.append(self.sampling.weight_key)
        if self.filter_zeros and self.weight_key not in variables:
            variables.append(self.weight_key)
        return tuple(variables)

    def sample(self, settings: SamplingSettings) -> EventSource:
        if not isinstance(settings, SamplingSettings):
            raise TypeError("settings must be a SamplingSettings instance")
        return replace(self, sampling=settings)

    @property
    def execution_source(self) -> EventSource:
        return self

    def transform(
        self,
        callback: Callable[[dict[str, np.ndarray]], Mapping[str, Any]],
        *,
        inputs: tuple[str, ...],
        schema: Mapping[str, DTypeLike],
        description: str | None = None,
    ) -> EventTransform:
        """Project row-preserving dictionary calculations after optional sampling."""
        return EventTransform(
            self, ChunkOperation(callback, schema, inputs, description)
        )

    def preprocess(
        self,
        callback: Callable[[dict[str, np.ndarray]], Mapping[str, Any]],
        *,
        schema: Mapping[str, DTypeLike],
        description: str | None = None,
    ) -> EventSource:
        """Derive columns/reject rows before sampling, preserving retained weights.

        The callback returns a complete aligned dictionary in retained input order.
        It must be deterministic and batch independent; no row expansion is allowed.
        Sampling-weight retention is checked when sampling settings are supplied.
        """
        if self.sampling is not None:
            raise ValueError("preprocess must be called before sample")
        stage = ChunkOperation(callback, schema, label=description)
        return replace(self, preprocessors=(*self.preprocessors, stage))

    @property
    def output_variables(self) -> tuple[str, ...]:
        """Projectable processed names; hidden raw weights are not implicit outputs."""
        return (
            tuple(self.preprocessors[-1].schema)
            if self.preprocessors
            else self.variables
        )

    def _processed_schema(self, raw: Mapping[str, np.dtype]) -> dict[str, np.dtype]:
        return dict(self.preprocessors[-1].schema if self.preprocessors else raw)

    def _inspect(
        self, context: SourceContext
    ) -> tuple[McStasNeXus, list[str], dict[str, np.dtype], int]:
        if not isinstance(context, SourceContext):
            raise TypeError("Event access requires a SourceContext")
        reader = context.mcstas
        if self.component_name is None:
            components = list(reader.get_components_with_ids())
        else:
            components = list(_names(self.component_name, "component_name"))
        dtype = None
        count = 0
        for component in components:
            info = reader.get_info_entry(component)
            tokens = reader.get_component_variables(component).rstrip(" ").split(" ")
            if any(not token or any(c.isspace() for c in token) for token in tokens):
                raise ValueError(
                    f"Component {component!r} variables must be space-separated tokens"
                )
            if len(set(tokens)) != len(tokens):
                raise ValueError(
                    f"Component {component!r} has duplicate event variables"
                )
            for variable in self.required_variables:
                if variable not in tokens:
                    raise ValueError(
                        f"Component {component!r} has no exact event variable "
                        f"{variable!r}"
                    )
            if "events" not in info or not isinstance(info["events"], h5py.Dataset):
                raise ValueError(f"Component {component!r} has no events dataset")
            events = info["events"]
            if (
                events.shape is None
                or events.ndim != 2
                or events.shape[1] != len(tokens)
            ):
                raise ValueError(
                    f"Component {component!r} events must be a 2D table with "
                    f"{len(tokens)} columns; got {events.shape}"
                )
            if events.dtype.kind not in "iufc":
                raise TypeError(
                    f"Component {component!r} event table must have a numeric dtype; "
                    f"got {events.dtype}"
                )
            table_dtype = events.dtype
            if self.sampling is not None and table_dtype.kind == "c":
                raise TypeError(
                    f"Component {component!r} sampling weights must be real"
                )
            if dtype is not None and dtype != table_dtype:
                raise ValueError(
                    f"Component {component!r} event dtype {table_dtype} differs from "
                    f"{dtype}; mixed source table dtypes are not supported"
                )
            dtype = table_dtype
            count += events.shape[0]
        if dtype is None:
            raise ValueError(
                "No event components selected; an event schema is unavailable"
            )
        return reader, components, dict.fromkeys(self.required_variables, dtype), count

    def schema(self, context: SourceContext) -> dict[str, np.dtype]:
        """Inspect final schemas without executing callbacks or reading rows."""
        return self._processed_schema(self.raw_schema(context))

    def raw_schema(self, context: SourceContext) -> dict[str, np.dtype]:
        """Inspect raw read dependencies, not derived preprocessing outputs."""
        return self._inspect(context)[2]

    def row_count(self, context: SourceContext) -> int | None:
        """Input rows before sampling; unknown when explicit filtering is enabled."""
        count = self._inspect(context)[3]
        return None if self.filter_zeros or self.preprocessors else count

    def iter_chunks(self, context: SourceContext) -> Iterator[dict[str, np.ndarray]]:
        reader, components, schema, _ = self._inspect(context)
        chunks = reader.iter_event_data(
            list(schema), component_name=components, chunk_size=self.read_chunk_size
        )
        while True:
            context._require_data()
            try:
                chunk = next(chunks)
            except StopIteration:
                return
            arrays = _event_arrays(chunk)
            if any(variable not in arrays for variable in schema):
                raise ValueError("Reader chunk is missing required event variables")
            if self.filter_zeros:
                nonzero = arrays[self.weight_key] != 0
                if not np.any(nonzero):
                    continue
                arrays = {key: values[nonzero] for key, values in arrays.items()}
            for index, stage in enumerate(self.preprocessors, 1):
                arrays = stage.apply(
                    arrays, allow_filter=True, context=f"Preprocessing stage {index}"
                )
            yield arrays

    def __getitem__(self, variable: str) -> EventColumn:
        return EventColumn(self, variable)

    @property
    def description(self) -> str:
        components = "components with pixel IDs"
        if self.component_name is not None:
            names = _names(self.component_name, "component_name")
            components = ", ".join(names[:3]) + (", ..." if len(names) > 3 else "")
        variables = ", ".join(self.variables[:5]) + (
            ", ..." if len(self.variables) > 5 else ""
        )
        result = (
            f"events from {components}; variables=({variables}); "
            f"read_chunk_size={self.read_chunk_size}; filter_zeros={self.filter_zeros}"
        )
        if self.filter_zeros:
            result += f" (weight_key={self.weight_key!r})"
        for index, stage in enumerate(self.preprocessors[:3], 1):
            result += f"; preprocessing {index}: {stage.description}"
        if len(self.preprocessors) > 3:
            result += f"; ... ({len(self.preprocessors) - 3} more preprocessing stages)"
        if self.sampling is not None:
            settings = self.sampling
            result += (
                f"; weighted sample requested={settings.n_samples}, "
                f"seed={settings.seed}, "
                f"ordered={settings.ordered}, weight_key={settings.weight_key!r}"
            )
        return result

    def __repr__(self) -> str:
        return f"EventSource({self.description})"


def event_source(
    *,
    component_name: str | list[str] | tuple[str, ...] | None = None,
    variables: str | list[str] | tuple[str, ...] = ("p", "t", "id"),
    read_chunk_size: int = 100_000,
    filter_zeros: bool = False,
    weight_key: str = "p",
) -> EventSource:
    """Describe shared event input without opening a file or reading any data."""
    return EventSource(
        component_name=tuple(component_name)
        if isinstance(component_name, list)
        else component_name,
        variables=_names(variables, "variables"),
        read_chunk_size=read_chunk_size,
        filter_zeros=filter_zeros,
        weight_key=weight_key,
    )


def _real_scalar(value: Any, name: str) -> int | float:
    array = np.asarray(value)
    if array.ndim != 0 or array.dtype.kind not in "iuf":
        raise TypeError(f"{name} must be a finite real numeric scalar")
    scalar = int(array) if array.dtype.kind in "iu" else float(array)
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    return scalar


def _checked_cast(values: Any, dtype: DTypeLike) -> np.ndarray:
    """Check range/integrality; a declared floating dtype permits precision loss."""
    dtype = _numeric_dtype(dtype)
    array = np.asarray(values)
    if array.dtype.kind == "O":
        if any(not isinstance(value, Number) for value in array.flat):
            raise TypeError("Column results must contain numeric values")
        if dtype.kind != "c" and any(
            isinstance(value, (complex, np.complexfloating)) for value in array.flat
        ):
            raise TypeError("Cannot cast complex column values to a real dtype")
    elif array.dtype.kind not in "iufc":
        raise TypeError("Column results must contain numeric values")
    if dtype.kind != "c" and array.dtype.kind == "c":
        raise TypeError("Cannot cast complex column values to a real dtype")
    if dtype.kind in "iu":
        limits = np.iinfo(dtype)
        if array.dtype.kind == "O":
            for value in array.flat:
                try:
                    integer = int(value)
                except (ValueError, OverflowError, TypeError) as error:
                    raise ValueError(
                        "Integer columns require finite integral values"
                    ) from error
                if integer != value:
                    raise ValueError(
                        "Integer columns require integral values, not fractional values"
                    )
                if not limits.min <= integer <= limits.max:
                    raise OverflowError(f"Column values are out of range for {dtype}")
        elif array.size:
            if array.dtype.kind == "f" and (
                not np.all(np.isfinite(array)) or np.any(array != np.trunc(array))
            ):
                raise ValueError("Integer columns require finite integral values")
            # Python integers avoid float64 rounding of int64/uint64 limits.
            if int(array.min()) < limits.min or int(array.max()) > limits.max:
                raise OverflowError(f"Column values are out of range for {dtype}")
    try:
        with np.errstate(over="raise", invalid="raise"):
            return array.astype(dtype, copy=False)
    except (FloatingPointError, OverflowError) as error:
        raise OverflowError(f"Column values overflow {dtype}") from error


# Dimension and decimal exponent relative to seconds/metres. No general unit algebra.
_UNITS = {
    "s": ("time", 0),
    "ms": ("time", -3),
    "us": ("time", -6),
    "ns": ("time", -9),
    "ps": ("time", -12),
    "m": ("length", 0),
    "cm": ("length", -2),
    "mm": ("length", -3),
    "um": ("length", -6),
    "nm": ("length", -9),
    "angstrom": ("length", -10),
    "A": ("length", -10),
}


@dataclass(frozen=True, repr=False)
class ColumnOperation:
    """Inspectable transform descriptor; parameters and declared dtype are immutable."""

    kind: str
    parameters: Mapping[str, Any]
    dtype: np.dtype | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"scale", "offset", "convert_units", "map_values", "map"}:
            raise ValueError(f"Unknown column operation {self.kind!r}")
        parameters = dict(self.parameters)
        if self.kind == "map_values":
            parameters["mapping"] = MappingProxyType(
                {
                    key: value.item()
                    if isinstance(value, (np.ndarray, np.generic))
                    else value
                    for key, value in parameters["mapping"].items()
                }
            )
        object.__setattr__(self, "parameters", MappingProxyType(parameters))
        if self.dtype is not None:
            object.__setattr__(self, "dtype", _numeric_dtype(self.dtype))
        elif self.kind in {"map", "map_values"}:
            raise TypeError("Mapping operations require an explicit numeric dtype")

    def output_dtype(self, input_dtype: DTypeLike) -> np.dtype:
        dtype = _numeric_dtype(input_dtype)
        if self.dtype is not None:
            return self.dtype
        value = self.parameters["value" if self.kind == "offset" else "factor"]
        inferred = np.result_type(dtype, np.asarray(value).dtype)
        if dtype.kind in "iu" and isinstance(value, int) and inferred.kind == "f":
            if dtype.kind == "u" and value >= 0:
                return np.result_type(dtype, np.min_scalar_type(value))
            raise TypeError(
                "Integer arithmetic needs an explicit dtype for this mixed range"
            )
        return inferred

    @property
    def description(self) -> str:
        params = self.parameters
        if self.kind == "scale":
            text = f"scale by {params['factor']}"
        elif self.kind == "offset":
            text = f"offset by {params['value']}"
        elif self.kind == "convert_units":
            text = (
                f"convert {params['from_unit']} to {params['to_unit']} "
                f"(multiply by {params['factor']:g})"
            )
        elif self.kind == "map_values":
            text = (
                f"value mapping ({len(params['mapping'])} entries, "
                f"missing={params['missing']})"
            )
        else:
            name = getattr(params["callback"], "__name__", "")
            text = "custom callback" + (
                f" {name}" if name and name != "<lambda>" else ""
            )
            if params["description"] is not None:
                text += f" ({params['description']})"
        return text + (f"; dtype={self.dtype}" if self.dtype is not None else "")

    def __repr__(self) -> str:
        return f"ColumnOperation({self.description})"


@dataclass(frozen=True, eq=False, repr=False)
class EventTransform:
    """A batch-local multi-column projection sharing its source's execution.

    Complete output schemas and input names are declared without callback probes.
    Callbacks preserve event order/count and are deterministic and batch independent.
    Unlike preprocessing, these calculations run after any sampling operation.
    """

    source: EventSource | EventTransform
    operation: ChunkOperation

    def __post_init__(self) -> None:
        if not isinstance(self.source, (EventSource, EventTransform)):
            raise TypeError("source must be an event source or chunk transform")
        if (
            not isinstance(self.operation, ChunkOperation)
            or self.operation.inputs is None
        ):
            raise TypeError("Chunk transforms require declared input variables")
        for name in self.operation.inputs:
            if name not in self.source.output_variables:
                raise KeyError(
                    f"Chunk transform input {name!r} is not declared by its source"
                )
        if self.sampling is not None:
            dtype = self.operation.schema.get(self.sampling.weight_key)
            if dtype is not None and dtype.kind not in "iuf":
                raise TypeError("Transformed sampled weights must be real unit weights")

    @property
    def execution_source(self) -> EventSource:
        return self.source.execution_source

    @property
    def sampling(self) -> SamplingSettings | None:
        return self.execution_source.sampling

    @property
    def output_variables(self) -> tuple[str, ...]:
        return tuple(self.operation.schema)

    @property
    def chunk_operations(self) -> tuple[ChunkOperation, ...]:
        previous = (
            self.source.chunk_operations
            if isinstance(self.source, EventTransform)
            else ()
        )
        return (*previous, self.operation)

    def _processed_schema(self, raw: Mapping[str, np.dtype]) -> dict[str, np.dtype]:
        return dict(self.operation.schema)

    def transform(
        self,
        callback: Callable[[dict[str, np.ndarray]], Mapping[str, Any]],
        *,
        inputs: tuple[str, ...],
        schema: Mapping[str, DTypeLike],
        description: str | None = None,
    ) -> EventTransform:
        return EventTransform(
            self, ChunkOperation(callback, schema, inputs, description)
        )

    def apply(
        self,
        chunk: Mapping[str, np.ndarray],
        *,
        cache: dict[EventTransform, dict[str, np.ndarray]] | None = None,
    ) -> dict[str, np.ndarray]:
        if cache is not None and self in cache:
            return cache[self]
        upstream = (
            self.source.apply(chunk, cache=cache)
            if isinstance(self.source, EventTransform)
            else chunk
        )
        arrays = self.operation.apply(upstream)
        if self.sampling is not None and self.sampling.weight_key in arrays:
            if not np.all(arrays[self.sampling.weight_key] == 1):
                raise ValueError(
                    "Chunk transforms of sampled weights must preserve unit weights"
                )
        if cache is not None:
            cache[self] = arrays
        return arrays

    def __getitem__(self, variable: str) -> EventColumn:
        return EventColumn(self, variable)

    @property
    def description(self) -> str:
        return f"chunk transform: {self.operation.description}"

    def __repr__(self) -> str:
        return f"EventTransform({self.description})"


@dataclass(frozen=True, repr=False)
class EventColumn:
    """One declared source variable and an ordered, row-preserving pipeline.

    Callback schema is explicit and never inferred by running user code. Callbacks
    receive a private copy of the column and must be deterministic, row-order and
    batch-boundary independent; only dimensionality and row count can be checked.
    """

    source: EventSource | EventTransform
    variable: str
    operations: tuple[ColumnOperation, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not isinstance(self.source, (EventSource, EventTransform)):
            raise TypeError("source must be an EventSource or EventTransform")
        if self.variable not in self.source.output_variables:
            raise KeyError(
                f"Variable {self.variable!r} is not declared by this EventSource"
            )
        object.__setattr__(self, "operations", tuple(self.operations))
        for operation in self.operations:
            if not isinstance(operation, ColumnOperation):
                raise TypeError("operations must contain ColumnOperation descriptors")
            if (
                self.source.sampling is not None
                and self.variable == self.source.sampling.weight_key
            ):
                identity = (
                    operation.kind in {"scale", "convert_units"}
                    and operation.parameters["factor"] == 1
                ) or (operation.kind == "offset" and operation.parameters["value"] == 0)
                if not identity:
                    raise ValueError(
                        "Transforms of sampled weights must preserve unit weights"
                    )

    def _append(
        self, kind: str, params: Mapping[str, Any], dtype: DTypeLike | None
    ) -> EventColumn:
        operation = ColumnOperation(
            kind, params, None if dtype is None else _numeric_dtype(dtype)
        )
        return EventColumn(self.source, self.variable, (*self.operations, operation))

    def scale(self, factor: float, dtype: DTypeLike | None = None) -> EventColumn:
        return self._append("scale", {"factor": _real_scalar(factor, "factor")}, dtype)

    def offset(self, value: float, dtype: DTypeLike | None = None) -> EventColumn:
        return self._append("offset", {"value": _real_scalar(value, "offset")}, dtype)

    def convert_units(
        self, from_unit: str, to_unit: str, dtype: DTypeLike | None = None
    ) -> EventColumn:
        """Convert s/ms/us/ns/ps or m/cm/mm/um/nm/angstrom (alias A).

        McStas normally records ``t`` in seconds, but no input units are inferred.
        Declaring a conversion does not set any destination HDF5 attributes.
        """
        if from_unit not in _UNITS or to_unit not in _UNITS:
            raise ValueError(
                f"Unsupported unit conversion {from_unit!r} to {to_unit!r}"
            )
        dimension, exponent = _UNITS[from_unit]
        target_dimension, target_exponent = _UNITS[to_unit]
        if dimension != target_dimension:
            raise ValueError(f"Unit dimension mismatch: {from_unit!r} to {to_unit!r}")
        if (
            self.target_unit is not None
            and _UNITS[self.target_unit] != _UNITS[from_unit]
        ):
            raise ValueError(
                f"Previous conversion declares {self.target_unit!r}, not {from_unit!r}"
            )
        return self._append(
            "convert_units",
            {
                "from_unit": from_unit,
                "to_unit": to_unit,
                "factor": 10.0 ** (exponent - target_exponent),
            },
            dtype,
        )

    def map_values(
        self, mapping: Mapping[Any, Any], dtype: DTypeLike, missing: str = "error"
    ) -> EventColumn:
        """Map exact numeric keys; never round input IDs or discard missing rows."""
        if not isinstance(mapping, Mapping):
            raise TypeError("mapping must be a mapping of numeric values")
        if missing not in {"error", "keep"}:
            raise ValueError("missing must be 'error' or 'keep'")
        if dtype is None:
            raise TypeError("map_values requires an explicit numeric dtype")
        dtype = _numeric_dtype(dtype)
        for key, value in mapping.items():
            _real_scalar(key, "mapping key")
            if np.ndim(value) != 0:
                raise ValueError("Mapping values must be numeric scalars")
            _checked_cast(np.asarray(value, dtype=object), dtype)
        return self._append(
            "map_values", {"mapping": mapping, "missing": missing}, dtype
        )

    def map(
        self,
        callback: Callable[[np.ndarray], Any],
        *,
        dtype: DTypeLike,
        description: str | None = None,
    ) -> EventColumn:
        if not callable(callback):
            raise TypeError("callback must be callable")
        if dtype is None:
            raise TypeError("map requires an explicit numeric dtype")
        if description is not None and not isinstance(description, str):
            raise TypeError("description must be a string or None")
        return self._append(
            "map", {"callback": callback, "description": description}, dtype
        )

    def output_dtype(self, input_dtype: DTypeLike) -> np.dtype:
        """Resolve schema without data access or callback execution."""
        dtype = _numeric_dtype(input_dtype)
        for operation in self.operations:
            dtype = operation.output_dtype(dtype)
        return dtype

    @property
    def execution_source(self) -> EventSource:
        return self.source.execution_source

    @property
    def chunk_operations(self) -> tuple[ChunkOperation, ...]:
        return (
            self.source.chunk_operations
            if isinstance(self.source, EventTransform)
            else ()
        )

    def apply(
        self,
        chunk_dict: Mapping[str, Any],
        *,
        cache: dict[EventTransform, dict[str, np.ndarray]] | None = None,
    ) -> np.ndarray:
        if isinstance(self.source, EventTransform):
            chunk_dict = self.source.apply(chunk_dict, cache=cache)
        arrays = _event_arrays(chunk_dict)
        values = arrays[self.variable]
        _numeric_dtype(values.dtype)
        count = len(values)
        for operation in self.operations:
            params = operation.parameters
            dtype = operation.output_dtype(values.dtype)
            if operation.kind == "map":
                result = np.asarray(params["callback"](values.copy()))
            elif operation.kind == "map_values":
                mapping = params["mapping"]
                result = np.empty(count, dtype=object)
                for index, value in enumerate(values):
                    if value in mapping:
                        result[index] = mapping[value]
                    elif params["missing"] == "keep":
                        result[index] = value
                    else:
                        raise ValueError(
                            f"Missing mapping for variable {self.variable!r}: {value!r}"
                        )
            else:
                value = params["value" if operation.kind == "offset" else "factor"]
                operand = values
                if values.dtype.kind in "iu" and (
                    isinstance(value, int) or float(value).is_integer()
                ):
                    # Python integers prevent NumPy's fixed-width arithmetic wrapping.
                    operand, value = values.astype(object), int(value)
                elif values.dtype.kind in "iu" and dtype.kind in "iu":
                    operand, value = values.astype(object), Fraction(value)
                else:
                    computation_dtype = np.result_type(
                        values.dtype,
                        np.asarray(value).dtype,
                        dtype if dtype.kind in "fc" else values.dtype,
                    )
                    operand = values.astype(computation_dtype, copy=False)
                with np.errstate(over="raise", invalid="raise"):
                    try:
                        result = (
                            operand + value
                            if operation.kind == "offset"
                            else operand * value
                        )
                    except FloatingPointError as error:
                        raise OverflowError(
                            f"Column {self.variable!r} arithmetic overflow"
                        ) from error
            if result.ndim != 1 or len(result) != count:
                raise ValueError(
                    f"Transform of {self.variable!r} must preserve "
                    "one-dimensional row count"
                )
            values = _checked_cast(result, dtype)
        return values

    @property
    def target_unit(self) -> str | None:
        unit = None
        for operation in self.operations:
            if operation.kind == "convert_units":
                unit = operation.parameters["to_unit"]
            elif operation.kind in {"map", "map_values"}:
                unit = None
        return unit

    @property
    def description(self) -> str:
        steps = [self.source.description, self.variable]
        steps.extend(operation.description for operation in self.operations)
        return " -> ".join(steps)

    def __repr__(self) -> str:
        return f"EventColumn({self.description})"
