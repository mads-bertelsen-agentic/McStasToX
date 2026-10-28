# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 McCode-dev contributors (https://github.com/mccode-dev)
"""Metadata-only instructions for a custom NeXus layout.

Construction and ``repr`` do not open the base file. Inspection uses short-lived,
read-only handles and never reads dataset payloads or follows external links.
Warnings are returned as diagnostics, not emitted through Python's warning
machinery. This module records instructions; it does not write HDF5 files.
"""

from __future__ import annotations

import math
import posixpath
from collections.abc import Iterator, Mapping, MutableMapping
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path
from typing import Any

import h5py
import numpy as np


@dataclass
class Binding:
    """A literal, source callback, streamed copy, or execution-result binding."""

    kind: str
    value: Any
    selection: Any = None
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class NodePlan:
    """Effective output metadata, with original paths for retained payloads.

    Aliases share ``object_key``, ``creation``, ``attributes``, and ``bindings``.
    A writer must not read ``base_path`` when a whole replacement is present.
    """

    path: str
    kind: str
    base_path: str | None
    object_key: Any
    creation: dict[str, Any] = field(default_factory=dict)
    attributes: dict[str, Binding] = field(default_factory=dict)
    bindings: list[Binding] = field(default_factory=list)
    link: h5py.SoftLink | h5py.ExternalLink | None = None


@dataclass(frozen=True)
class Diagnostic:
    severity: str
    path: str
    message: str


@dataclass(frozen=True)
class ValidationReport:
    diagnostics: tuple[Diagnostic, ...] = ()

    @property
    def errors(self) -> tuple[Diagnostic, ...]:
        return tuple(d for d in self.diagnostics if d.severity == "error")

    @property
    def warnings(self) -> tuple[Diagnostic, ...]:
        return tuple(d for d in self.diagnostics if d.severity == "warning")

    def raise_if_errors(self) -> None:
        if self.errors:
            raise ValueError(
                "Invalid NeXus template:\n"
                + "\n".join(f"{d.path}: {d.message}" for d in self.errors)
            )


@dataclass(frozen=True)
class RetainedDataset:
    """One inherited object; aliases exclude its primary output ``path``.

    Byte counts describe the *original* dataset, even for partial updates.
    Allocated storage is not an estimate of the resulting output file size.
    """

    path: str
    aliases: tuple[str, ...]
    base_file: Path
    base_path: str
    shape: tuple[int, ...] | None
    dtype: np.dtype
    logical_bytes: int | None
    storage_bytes: int | None
    partial: bool
    unknown_reason: str | None = None

    @property
    def origin_file(self) -> Path:
        return self.base_file

    @property
    def origin_path(self) -> str:
        return self.base_path


@dataclass(eq=False)
class _Node:
    kind: str
    key: Any
    base_path: str | None = None
    creation: dict[str, Any] = field(default_factory=dict)
    attributes: dict[str, Binding] = field(default_factory=dict)
    bindings: list[Binding] = field(default_factory=list)
    link: h5py.SoftLink | h5py.ExternalLink | None = None
    children: dict[str, _Node | str] | None = None
    original_shape: tuple[int, ...] | None = None
    original_dtype: np.dtype | None = None
    logical_bytes: int | None = None
    storage_bytes: int | None = None
    unknown_reason: str | None = None
    features: tuple[str, ...] = ()
    edits: list[tuple[str, str, Any]] = field(default_factory=list)
    explicit_creation: set[str] = field(default_factory=set)
    description: str | None = None
    adaptations: list[str] = field(default_factory=list)
    issues: dict[str, str] = field(default_factory=dict)


_CREATION_KEYS = {
    "shape",
    "dtype",
    "maxshape",
    "chunks",
    "compression",
    "compression_opts",
    "shuffle",
    "fletcher32",
    "scaleoffset",
    "fillvalue",
    "track_times",
    "track_order",
    "external",
}
_RESULT_FIELDS = {
    "n_written",
    "n_selected",
    "total_weight",
    "total_weight_variance",
    "effective_duration",
    "effective_duration_variance",
}


def _path(path: str) -> str:
    if not isinstance(path, str):
        raise TypeError("HDF5 paths must be strings")
    if not path.startswith("/") or "\0" in path or ".." in path.split("/"):
        raise ValueError("Use a root-absolute HDF5 path without '..' or NUL")
    return "/" + "/".join(p for p in path.split("/") if p and p != ".")


def _shape(value: Any, *, unlimited: bool = False) -> tuple:
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        value = (value,)
    if not isinstance(value, (tuple, list)):
        raise TypeError("shape must be an integer or a sequence of dimensions")
    result = []
    for dim in value:
        if unlimited and dim is None:
            result.append(None)
        elif (
            isinstance(dim, (int, np.integer))
            and not isinstance(dim, bool)
            and dim >= 0
        ):
            result.append(int(dim))
        else:
            raise ValueError(
                "dimensions must be nonnegative integers (or None in maxshape)"
            )
    return tuple(result)


def _dtype_has(dtype: np.dtype, predicate: Any) -> bool:
    if predicate(dtype):
        return True
    if dtype.subdtype is not None:
        return _dtype_has(dtype.subdtype[0], predicate)
    return bool(dtype.fields) and any(
        _dtype_has(spec[0], predicate) for spec in dtype.fields.values()
    )


def _has_references(value: Any) -> bool:
    if isinstance(value, h5py.Reference):
        return True
    if isinstance(value, (np.ndarray, np.generic)):
        if _dtype_has(value.dtype, lambda d: h5py.check_dtype(ref=d) is not None):
            return True
        if value.dtype.kind == "O":
            return any(_has_references(v) for v in value.flat)
    if isinstance(value, (tuple, list)):
        return any(_has_references(v) for v in value)
    return False


def _literal_spec(value: Any) -> tuple[tuple[int, ...], np.dtype]:
    if isinstance(value, (h5py.Dataset, h5py.Group)):
        raise TypeError("live HDF5 objects are not literals; use copy_from with a path")
    array = np.asarray(value)
    dtype = array.dtype
    if dtype.kind == "U" or (
        dtype.kind == "O"
        and dtype.metadata is None
        and array.size > 0
        and all(isinstance(item, (str, bytes)) for item in array.flat)
    ):
        dtype = h5py.string_dtype("utf-8")
    return array.shape, np.dtype(dtype)


def _selection(
    shape: tuple | None, selection: Any
) -> tuple[tuple | None, tuple[range, ...]]:
    """Normalize basic indexing; an unknown shape permits syntax checks only."""
    items = selection if isinstance(selection, tuple) else (selection,)
    if sum(item is Ellipsis for item in items) > 1:
        raise ValueError("selection contains more than one ellipsis")
    if shape is None:
        for item in items:
            if item is Ellipsis or (
                isinstance(item, (int, np.integer)) and not isinstance(item, bool)
            ):
                continue
            if isinstance(item, slice):
                if item.step is not None and item.step <= 0:
                    raise ValueError("selection slice steps must be positive")
                item.indices(0)
            else:
                raise ValueError(
                    "only integers, slices, tuples, and ellipsis are supported"
                )
        return None, ()
    explicit = sum(item is not Ellipsis for item in items)
    if explicit > len(shape):
        raise ValueError("selection has too many indices")
    normalized = []
    for item in items:
        if item is Ellipsis:
            normalized.extend([slice(None)] * (len(shape) - explicit))
        else:
            normalized.append(item)
    normalized.extend([slice(None)] * (len(shape) - len(normalized)))
    selected_shape, regions = [], []
    for dim, item in zip(shape, normalized, strict=True):
        if isinstance(item, (int, np.integer)) and not isinstance(item, bool):
            index = int(item) + (dim if item < 0 else 0)
            if index < 0 or index >= dim:
                raise ValueError("selection index is out of bounds")
            regions.append(range(index, index + 1))
        elif isinstance(item, slice):
            if item.step is not None and item.step <= 0:
                raise ValueError("selection slice steps must be positive")
            region = range(*item.indices(dim))
            regions.append(region)
            selected_shape.append(len(region))
        else:
            raise ValueError(
                "only integers, slices, tuples, and ellipsis are supported"
            )
    return tuple(selected_shape), tuple(regions)


def _equivalent(a: Any, b: Any) -> bool:
    if a is b:
        return True
    if isinstance(a, Binding) and isinstance(b, Binding):
        return (
            a.kind == b.kind
            and _equivalent(a.selection, b.selection)
            and _equivalent(a.value, b.value)
            and _equivalent(a.options, b.options)
        )
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        return a.keys() == b.keys() and all(_equivalent(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
        return len(a) == len(b) and all(
            _equivalent(x, y) for x, y in zip(a, b, strict=True)
        )
    if isinstance(a, np.dtype) and isinstance(b, np.dtype):
        return a == b and a.metadata == b.metadata
    if isinstance(a, (np.ndarray, np.generic)) or isinstance(
        b, (np.ndarray, np.generic)
    ):
        return np.array_equal(a, b)
    if isinstance(a, (str, bytes, int, float, slice, type(None))) and type(a) is type(
        b
    ):
        return a == b
    return False


def _overlap(a: Binding, b: Binding, shape: tuple | None) -> bool:
    if a.selection is None or b.selection is None or shape is None:
        return True
    try:
        _, left = _selection(shape, a.selection)
        _, right = _selection(shape, b.selection)
    except (TypeError, ValueError):
        return True
    for x, y in zip(left, right, strict=True):
        if not x or not y or x[-1] < y[0] or y[-1] < x[0]:
            return False
        # Bounding ranges are conservative for strided updates.
    return True


def _brief(value: Any, limit: int = 150, _depth: int = 0) -> str:
    """Bound both traversal and rendering of user-provided metadata."""
    if isinstance(value, str):
        text = value[:limit]
        return text + ("..." if len(value) > limit else "")
    if isinstance(value, bytes):
        return repr(value[:limit]) + ("..." if len(value) > limit else "")
    if isinstance(value, np.ndarray):
        return f"array(shape={value.shape}, dtype={value.dtype})"
    if isinstance(value, Mapping):
        if _depth >= 2:
            return f"mapping ({len(value)} entries)"
        entries = [
            f"{_brief(k, 30, _depth + 1)}: {_brief(v, 45, _depth + 1)}"
            for k, v in islice(value.items(), 3)
        ]
        if len(value) > 3:
            entries.append(f"... ({len(value)} entries)")
        return _brief("{" + ", ".join(entries) + "}", limit)
    if isinstance(value, (tuple, list)):
        if _depth >= 2:
            return f"sequence ({len(value)} items)"
        text = ", ".join(_brief(v, 35, _depth + 1) for v in islice(value, 3))
        if len(value) > 3:
            text += f", ... ({len(value)} items)"
        return _brief("[" + text + "]", limit)
    if callable(value):
        name = getattr(value, "__name__", "")
        return "custom callback" + (f" {name}" if name and name != "<lambda>" else "")
    if value is None or isinstance(
        value, (bool, int, float, np.generic, np.dtype, slice)
    ):
        return str(value)[:limit]
    return type(value).__name__


class AttributeProxy:
    """An attribute instruction, not a live h5py attribute handle."""

    def __init__(self, node: NodeProxy, name: str):
        self.node = node
        self.name = name

    @property
    def binding(self) -> Binding:
        return self.node._node.attributes[self.name]

    def _set(self, binding: Binding) -> AttributeProxy:
        node = self.node._node
        self.node._template._edit(
            node, self.node.path, f"attribute:{self.name}", binding
        )
        node.issues.pop(f"attribute:{self.name}", None)
        node.attributes[self.name] = binding
        return self

    def set_value(
        self, value: Any, *, description: str | None = None
    ) -> AttributeProxy:
        if isinstance(value, (h5py.Dataset, h5py.Group)):
            raise TypeError(
                "live HDF5 objects are not literals; use copy_from with a path"
            )
        return self._set(Binding("literal", value, options=_options(description)))

    def set_from(
        self, callback: Any, *, description: str | None = None
    ) -> AttributeProxy:
        if not callable(callback):
            raise TypeError("set_from requires a callable(source_context)")
        return self._set(Binding("callback", callback, options=_options(description)))

    def set_from_result(
        self, source: Any, selector: Any, *, description: str | None = None
    ) -> AttributeProxy:
        _result_selector(selector)
        return self._set(
            Binding("result", (source, selector), options=_options(description))
        )

    def __repr__(self) -> str:
        return f"AttributeProxy({self.node.path!r}, {self.name!r})"


class AttributeMap(MutableMapping[str, AttributeProxy]):
    def __init__(self, node: NodeProxy):
        self.node = node

    def __getitem__(self, name: str) -> AttributeProxy:
        if not isinstance(name, str) or not name or "\0" in name:
            raise ValueError("attribute names must be nonempty strings without NUL")
        # Missing attributes can be bound using attrs[name].set_from(...).
        return AttributeProxy(self.node, name)

    def __setitem__(self, name: str, value: Any) -> None:
        self[name].set_value(value)

    def __delitem__(self, name: str) -> None:
        node = self.node._node
        if name not in node.attributes:
            raise KeyError(name)
        self.node._template._edit(node, self.node.path, f"attribute:{name}", None)
        node.issues.pop(f"attribute:{name}", None)
        del node.attributes[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self.node._node.attributes)

    def __len__(self) -> int:
        return len(self.node._node.attributes)

    def __contains__(self, name: object) -> bool:
        return name in self.node._node.attributes


def _options(description: str | None, **values: Any) -> dict[str, Any]:
    if description is not None:
        if not isinstance(description, str):
            raise TypeError("description must be a string")
        values["description"] = description
    return {k: v for k, v in values.items() if v is not None}


def _result_selector(selector: Any) -> None:
    if not callable(selector) and (not isinstance(selector, str) or not selector):
        raise TypeError("result selector must be a field name or callable(result)")


class NodeProxy:
    """A path-based group/dataset proxy. Fluent editing methods return self."""

    def __init__(self, template: NexusTemplate, path: str):
        self._template = template
        self.path = path

    @property
    def _node(self) -> _Node:
        return self._template._lookup(self.path)

    @property
    def kind(self) -> str:
        return self._node.kind

    @property
    def attrs(self) -> AttributeMap:
        if self.kind not in {"group", "dataset"}:
            raise TypeError(
                "links do not have attributes; edit their hard-linked target"
            )
        return AttributeMap(self)

    @property
    def creation(self) -> dict[str, Any]:
        return dict(self._node.creation)

    def _child_path(self, path: str) -> str:
        return _path(path if path.startswith("/") else posixpath.join(self.path, path))

    def __getitem__(self, path: str) -> NodeProxy:
        return self._template[self._child_path(path)]

    def __delitem__(self, path: str) -> None:
        del self._template[self._child_path(path)]

    def add_group(self, path: str) -> NodeProxy:
        return self._template.add_group(self._child_path(path))

    def add_dataset(self, path: str, **kwargs: Any) -> NodeProxy:
        return self._template.add_dataset(self._child_path(path), **kwargs)

    def describe(self, description: str) -> NodeProxy:
        _options(description)
        self._node.description = description
        return self

    def _bind(self, binding: Binding, creation: dict[str, Any]) -> NodeProxy:
        node = self._node
        if node.kind != "dataset":
            raise TypeError(f"{self.path} is not a dataset")
        spec = self._template._creation(creation)
        if binding.kind == "literal":
            shape, dtype = _literal_spec(binding.value)
            if binding.selection is None:
                spec.setdefault("shape", shape)
                if "dtype" not in spec and "dtype" in node.creation:
                    inherited = h5py.check_string_dtype(node.creation["dtype"])
                    literal = h5py.check_string_dtype(dtype)
                    if (
                        inherited is not None
                        and inherited.encoding == "utf-8"
                        and literal is not None
                    ):
                        # Preserve encoding without retaining an undersized width.
                        dtype = h5py.string_dtype("utf-8", literal.length)
                spec.setdefault("dtype", dtype)
        if binding.selection is None:
            dtype = spec.get("dtype", node.creation.get("dtype"))
            if (
                binding.kind != "literal"
                and dtype is not None
                and "dtype" not in spec
                and "dtype" not in node.explicit_creation
                and (
                    dtype.subdtype is not None
                    or _dtype_has(
                        dtype,
                        lambda d: d.kind == "O" and h5py.check_string_dtype(d) is None,
                    )
                )
            ):
                node.creation.pop("dtype", None)
                node.creation.pop("fillvalue", None)
                node.adaptations.append(
                    "Unsupported inherited dtype/fillvalue omitted for whole "
                    "replacement; output dtype must be resolved by the writer"
                )
            elif (
                "fillvalue" in node.creation
                and "fillvalue" not in spec
                and "fillvalue" not in node.explicit_creation
                and dtype is not None
            ):
                try:
                    self._template._validate_literal(
                        node.creation["fillvalue"], (), dtype, False, False
                    )
                except (TypeError, ValueError, OverflowError):
                    node.creation.pop("fillvalue")
                    node.adaptations.append(
                        "Reset incompatible inherited fillvalue "
                        "after whole dtype replacement"
                    )
        for name, value in spec.items():
            self._template._edit(node, self.path, f"creation:{name}", value)
        node.creation.update(spec)
        node.explicit_creation.update(creation)
        self._template._edit(node, self.path, "binding", binding)
        if binding.selection is None:
            node.bindings.clear()
        node.bindings.append(binding)
        return self

    def set_value(
        self,
        value: Any,
        selection: Any = None,
        *,
        description: str | None = None,
        **creation: Any,
    ) -> NodeProxy:
        return self._bind(
            Binding("literal", value, selection, _options(description)), creation
        )

    def set_from(
        self,
        callback: Any,
        selection: Any = None,
        *,
        description: str | None = None,
        **creation: Any,
    ) -> NodeProxy:
        if not callable(callback):
            raise TypeError("set_from requires a callable(source_context)")
        return self._bind(
            Binding("callback", callback, selection, _options(description)), creation
        )

    def copy_from(
        self,
        source: Any,
        selection: Any = None,
        *,
        read_chunk_size: int | None = None,
        write_chunk_size: int | None = None,
        columns: Any = None,
        description: str | None = None,
        **creation: Any,
    ) -> NodeProxy:
        if isinstance(source, str):
            source = _path(source)
        elif source is None or isinstance(source, (h5py.Dataset, h5py.Group, Iterator)):
            raise TypeError(
                "copy_from requires an absolute path or a provider specification"
            )
        elif not any(
            hasattr(source, name) for name in ("description", "source", "variables")
        ):
            raise TypeError(
                "copy_from requires an absolute path or a provider specification"
            )
        if isinstance(columns, str):
            raise TypeError("columns must be a sequence of column names, not a string")
        return self._bind(
            Binding(
                "copy",
                source,
                selection,
                _options(
                    description,
                    read_chunk_size=read_chunk_size,
                    write_chunk_size=write_chunk_size,
                    columns=tuple(columns) if columns is not None else None,
                ),
            ),
            creation,
        )

    def set_from_result(
        self,
        source: Any,
        selector: Any,
        selection: Any = None,
        *,
        description: str | None = None,
        **creation: Any,
    ) -> NodeProxy:
        _result_selector(selector)
        return self._bind(
            Binding("result", (source, selector), selection, _options(description)),
            creation,
        )

    def __repr__(self) -> str:
        return f"NodeProxy({self.path!r})"


_UNSET = object()


class NexusTemplate:
    """A lazy, reusable instruction graph, optionally backed by a base file.

    The base file must remain unchanged after the first metadata inspection.
    No handles or live h5py objects are kept in this graph. Hard-link aliases
    deliberately remain shared, including after whole dataset replacement.
    """

    def __init__(self, base_path: str | Path | None = None):
        self.base_path = (
            Path(base_path).expanduser().absolute() if base_path is not None else None
        )
        self._objects: dict[Any, _Node] = {}
        self._next_key = 0
        self._root: _Node | None = None
        self._diagnostics: list[Diagnostic] = []
        self._removals: list[str] = []
        self._moves: list[tuple[str, str]] = []
        self._repairs: list[tuple[str, str]] = []
        if self.base_path is None:
            self._root = self._new_node("group")

    def _new_node(self, kind: str, **kwargs: Any) -> _Node:
        self._next_key += 1
        node = _Node(kind, ("new", self._next_key), **kwargs)
        if kind == "group":
            node.children = {}
        self._objects[node.key] = node
        return node

    def _load(self, path: str) -> _Node:
        with h5py.File(self.base_path, "r") as file:
            link = file.get(path, getlink=True) if path != "/" else None
            if isinstance(link, (h5py.SoftLink, h5py.ExternalLink)):
                return self._new_node(
                    "soft" if isinstance(link, h5py.SoftLink) else "external",
                    base_path=path,
                    link=link,
                )
            obj = file[path]
            if not isinstance(obj, (h5py.Group, h5py.Dataset)):
                raise TypeError(
                    f"Unsupported HDF5 object {type(obj).__name__} at {path}; "
                    "remove this node before writing"
                )
            key = h5py.h5o.get_info(obj.id).addr
            if key in self._objects:
                return self._objects[key]
            node = _Node(
                "group" if isinstance(obj, h5py.Group) else "dataset", key, path
            )
            for name, value in obj.attrs.items():
                if isinstance(value, np.ndarray):
                    dtype = value.dtype
                    if dtype.kind == "O" and dtype.metadata is None:
                        # h5py can omit string metadata, even for empty attributes.
                        original_dtype = obj.attrs.get_id(name).dtype
                        if h5py.check_string_dtype(original_dtype) is not None:
                            dtype = original_dtype
                    copied = np.empty(value.shape, dtype=dtype)
                    copied[...] = value
                    value = copied
                node.attributes[name] = Binding("literal", value)
            if node.kind == "dataset":
                node.original_shape, node.original_dtype = obj.shape, obj.dtype
                node.creation = {"shape": obj.shape, "dtype": obj.dtype}
                for name in (
                    "maxshape",
                    "chunks",
                    "compression",
                    "compression_opts",
                    "shuffle",
                    "fletcher32",
                    "scaleoffset",
                ):
                    value = getattr(obj, name)
                    # Contiguous datasets' maxshape == shape is not a resize request.
                    if value is not None and not (
                        name == "maxshape" and obj.chunks is None
                    ):
                        node.creation[name] = value
                if not _dtype_has(
                    obj.dtype, lambda d: h5py.check_dtype(ref=d) is not None
                ):
                    node.creation["fillvalue"] = obj.fillvalue
                properties = obj.id.get_create_plist()
                node.creation["track_times"] = bool(properties.get_obj_track_times())
                node.creation["track_order"] = bool(
                    properties.get_attr_creation_order() & h5py.h5p.CRT_ORDER_TRACKED
                )
                features = []
                if obj.is_virtual:
                    features.append("virtual dataset")
                if obj.external:
                    features.append("external raw storage")
                if obj.shape is None:
                    features.append("null dataspace")
                node.features = tuple(features)
                variable = _dtype_has(
                    obj.dtype, lambda d: h5py.check_dtype(vlen=d) is not None
                )
                if variable or features:
                    node.unknown_reason = (
                        ", ".join(features)
                        if features
                        else "variable-length payload; heap storage is not included"
                    )
                else:
                    node.logical_bytes = math.prod(obj.shape) * obj.dtype.itemsize
                if not features:
                    node.storage_bytes = int(obj.id.get_storage_size())
            self._objects[key] = node
            return node

    def _children(self, node: _Node) -> dict[str, _Node | str]:
        if node.kind != "group":
            raise TypeError("parent is not a group (links are not followed implicitly)")
        if node.children is None:
            with h5py.File(self.base_path, "r") as file:
                group = file[node.base_path]
                # These remain original base paths even after a parent is moved.
                node.children = {
                    name: posixpath.join(node.base_path, name) for name in group
                }
        return node.children

    def _child(self, parent: _Node, name: str) -> _Node:
        children = self._children(parent)
        child = children[name]
        if isinstance(child, str):
            child = self._load(child)
            children[name] = child
        return child

    def _lookup(self, path: str) -> _Node:
        if self._root is None:
            self._root = self._load("/")
        node = self._root
        for name in path.strip("/").split("/") if path != "/" else ():
            node = self._child(node, name)
        return node

    def __getitem__(self, path: str) -> NodeProxy:
        path = _path(path)
        self._lookup(path)
        return NodeProxy(self, path)

    @property
    def attrs(self) -> AttributeMap:
        return self["/"].attrs

    def __contains__(self, path: object) -> bool:
        if not isinstance(path, str):
            return False
        try:
            self._lookup(_path(path))
        except (KeyError, TypeError):
            return False
        return True

    def _parent(self, path: str) -> tuple[_Node, str]:
        if path == "/":
            raise ValueError("the root cannot be added, deleted, or moved")
        parent_path, name = posixpath.split(path)
        parent = self._lookup(parent_path or "/")
        self._children(parent)
        return parent, name

    def _insert(self, path: str, node: _Node) -> NodeProxy:
        parent, name = self._parent(path)
        children = self._children(parent)
        if name in children:
            raise ValueError(f"a node already exists at {path}")
        children[name] = node
        self._edit(parent, posixpath.dirname(path) or "/", f"child:{name}", node.key)
        return NodeProxy(self, path)

    def add_group(self, path: str) -> NodeProxy:
        path = _path(path)
        self._parent(path)
        return self._insert(path, self._new_node("group"))

    @staticmethod
    def _creation(values: dict[str, Any]) -> dict[str, Any]:
        if any(
            isinstance(value, (h5py.Dataset, h5py.Group)) for value in values.values()
        ):
            raise TypeError("live HDF5 objects are not dataset creation properties")
        unknown = values.keys() - _CREATION_KEYS
        if unknown:
            raise TypeError(
                f"unknown dataset creation properties: {', '.join(sorted(unknown))}"
            )
        result = dict(values)
        for name in ("shape", "maxshape"):
            if name in result and result[name] is not None:
                result[name] = _shape(result[name], unlimited=name == "maxshape")
        if "dtype" in result:
            dtype = np.dtype(result["dtype"])
            result["dtype"] = h5py.string_dtype("utf-8") if dtype.kind == "U" else dtype
        if isinstance(result.get("chunks"), (tuple, list)):
            result["chunks"] = _shape(result["chunks"])
        return result

    def add_dataset(self, path: str, value: Any = _UNSET, **creation: Any) -> NodeProxy:
        path = _path(path)
        spec = self._creation(creation)
        parent, name = self._parent(path)
        if name in self._children(parent):
            raise ValueError(f"a node already exists at {path}")
        if value is not _UNSET:
            shape, dtype = _literal_spec(value)
            spec.setdefault("shape", shape)
            spec.setdefault("dtype", dtype)
        elif "shape" in spec:
            spec.setdefault("dtype", np.dtype("float32"))
        node = self._new_node("dataset", creation=spec, explicit_creation=set(creation))
        if value is not _UNSET:
            node.bindings.append(Binding("literal", value))
        return self._insert(path, node)

    def __delitem__(self, path: str) -> None:
        path = _path(path)
        parent, name = self._parent(path)
        children = self._children(parent)
        if name not in children:
            raise KeyError(path)
        del children[name]
        self._removals.append(path)
        self._edit(parent, posixpath.dirname(path) or "/", f"child:{name}", None)

    def create_link(
        self, path: str, target: str | h5py.SoftLink | h5py.ExternalLink
    ) -> NodeProxy:
        """Add an explicit hard alias (a path string), soft link, or external link."""
        path = _path(path)
        if isinstance(target, str):
            node = self._lookup(_path(target))
            if node.kind == "group":
                parent, _ = self._parent(path)
                if self._reaches(node, parent):
                    raise ValueError("hard group links must not create a cycle")
            if node.kind not in {"group", "dataset"}:
                raise TypeError("hard links must target a group or dataset")
        elif isinstance(target, (h5py.SoftLink, h5py.ExternalLink)):
            node = self._new_node(
                "soft" if isinstance(target, h5py.SoftLink) else "external", link=target
            )
        else:
            raise TypeError(
                "target must be a hard-link path, SoftLink, or ExternalLink"
            )
        return self._insert(path, node)

    def _reaches(self, node: _Node, target: _Node, seen: set | None = None) -> bool:
        if node is target:
            return True
        if node.kind != "group":
            return False
        seen = set() if seen is None else seen
        if node.key in seen:
            return False
        seen.add(node.key)
        return any(
            self._reaches(self._child(node, n), target, seen)
            for n in self._children(node)
        )

    def move(
        self, old_path: str, new_path: str, *, repair_links: bool = False
    ) -> NodeProxy:
        old_path, new_path = _path(old_path), _path(new_path)
        old_parent, old_name = self._parent(old_path)
        node = self._child(old_parent, old_name)
        new_parent, new_name = self._parent(new_path)
        if new_name in self._children(new_parent):
            raise ValueError(f"a node already exists at {new_path}")
        if node.kind == "group" and self._reaches(node, new_parent):
            raise ValueError("a group cannot be moved into itself or a descendant")
        before = self._snapshot() if repair_links else None
        if before is not None and any(
            sum(p.object_key == parent.key for p in before.values()) > 1
            for parent in (old_parent, new_parent)
        ):
            node.issues["move"] = (
                "Link repair for moves through shared parent-group aliases "
                "is unsupported; remove the parent alias before moving"
            )
        del self._children(old_parent)[old_name]
        self._children(new_parent)[new_name] = node
        self._moves.append((old_path, new_path))
        self._edit(
            old_parent, posixpath.dirname(old_path) or "/", f"child:{old_name}", None
        )
        self._edit(
            new_parent,
            posixpath.dirname(new_path) or "/",
            f"child:{new_name}",
            node.key,
        )
        if before is not None:
            self._repair(before, old_path, new_path, relocating=True)
        else:
            self._diagnostics.append(
                Diagnostic(
                    "warning",
                    new_path,
                    f"Moved from {old_path} without link repair; inspect soft links "
                    "and NeXus target attributes",
                )
            )
        return NodeProxy(self, new_path)

    @staticmethod
    def _mapped(path: str, old: str, new: str) -> str:
        return (
            new + path[len(old) :]
            if path == old or path.startswith(old + "/")
            else path
        )

    def repair_links(self, old_path: str, new_path: str) -> NexusTemplate:
        """Repair incoming paths; use move(repair_links=True) for relocation."""
        old_path, new_path = _path(old_path), _path(new_path)
        if old_path == "/":
            raise ValueError("repairing the root path is not supported")
        self._lookup(new_path)
        self._repair(self._snapshot(), old_path, new_path, relocating=False)
        return self

    def _repair(
        self, snapshot: dict[str, NodePlan], old: str, new: str, *, relocating: bool
    ) -> None:
        soft_links: dict[Any, list[tuple[str, str, str, bool]]] = {}
        repaired_attrs = set()
        for path, plan in snapshot.items():
            node = self._objects[plan.object_key]
            if plan.kind == "soft":
                target = posixpath.normpath(
                    posixpath.join(posixpath.dirname(path), plan.link.path)
                )
                mapped = self._mapped(target, old, new)
                owner_moved = relocating and self._mapped(path, old, new) != path
                needs_repair = mapped != target or (
                    owner_moved and not plan.link.path.startswith("/")
                )
                soft_links.setdefault(node.key, []).append(
                    (path, target, mapped, needs_repair)
                )
            if node.key in repaired_attrs:
                continue
            repaired_attrs.add(node.key)
            binding = node.attributes.get("target")
            if binding is not None and binding.kind == "literal":
                value = binding.value
                array = value if isinstance(value, np.ndarray) else None
                if array is not None and array.size == 1:
                    value = array.flat[0]
                if isinstance(value, (str, bytes)):
                    try:
                        text = (
                            value.decode("utf-8") if isinstance(value, bytes) else value
                        )
                    except UnicodeDecodeError:
                        node.issues["attribute:target"] = (
                            "Cannot repair non-UTF-8 target metadata; "
                            "replace the attribute explicitly"
                        )
                        continue
                    if text.startswith("/"):
                        mapped = self._mapped(text, old, new)
                        if mapped != text:
                            value = (
                                mapped.encode("utf-8")
                                if isinstance(value, bytes)
                                else mapped
                            )
                            if array is not None:
                                dtype = (
                                    h5py.string_dtype("utf-8")
                                    if array.dtype.kind != "S"
                                    else None
                                )
                                value = np.asarray(value, dtype=dtype).reshape(
                                    array.shape
                                )
                            node.attributes["target"] = Binding(
                                "literal",
                                value,
                                options=dict(binding.options),
                            )
                            self._edit(node, path, "repair:target", mapped)
                elif array is not None:
                    node.issues["attribute:target"] = (
                        "Only scalar or length-one NeXus target attributes "
                        "can be repaired; replace this attribute explicitly"
                    )
        for key, contexts in soft_links.items():
            if not any(context[3] for context in contexts):
                continue
            path = contexts[0][0]
            node = self._objects[key]
            if len({context[2] for context in contexts}) > 1:
                node.issues["link"] = (
                    "Soft-link repair is ambiguous through shared group aliases; "
                    "replace the link explicitly"
                )
            elif any(
                target not in snapshot and mapped == target
                for _, target, mapped, _ in contexts
            ):
                node.issues["link"] = (
                    "Cannot safely repair an unresolved relative soft link "
                    "in a moved group; replace this link explicitly"
                )
            else:
                node.link = h5py.SoftLink(contexts[0][2])
                self._edit(node, path, "repair:link", contexts[0][2])
        self._repairs.append((old, new))

    def _edit(self, node: _Node, path: str, subject: str, value: Any) -> None:
        node.edits.append((path, subject, value))

    def _snapshot(self) -> dict[str, NodePlan]:
        """Return the effective metadata tree; no dataset payloads are read.

        A group cycle is represented by its alias node but not expanded further.
        Validation rejects such cycles; ordinary acyclic aliases are expanded.
        """
        result = {}

        def visit(node: _Node, path: str, ancestors: set) -> None:
            result[path] = NodePlan(
                path,
                node.kind,
                node.base_path,
                node.key,
                node.creation,
                node.attributes,
                node.bindings,
                node.link,
            )
            if node.kind == "group" and node.key not in ancestors:
                ancestors = ancestors | {node.key}
                for name in sorted(self._children(node)):
                    visit(
                        self._child(node, name), posixpath.join(path, name), ancestors
                    )

        visit(self._lookup("/"), "/", set())
        return result

    def _retained(
        self, snapshot: dict[str, NodePlan], min_bytes: int = 0
    ) -> list[RetainedDataset]:
        paths: dict[Any, list[str]] = {}
        for path, plan in snapshot.items():
            if (
                plan.kind == "dataset"
                and plan.base_path is not None
                and not any(b.selection is None for b in plan.bindings)
            ):
                paths.setdefault(plan.object_key, []).append(path)
        records = []
        for key, aliases in paths.items():
            node = self._objects[key]
            if node.logical_bytes is not None and node.logical_bytes < min_bytes:
                continue
            records.append(
                RetainedDataset(
                    aliases[0],
                    tuple(aliases[1:]),
                    self.base_path,
                    node.base_path,
                    node.original_shape,
                    node.original_dtype,
                    node.logical_bytes,
                    node.storage_bytes,
                    bool(node.bindings),
                    node.unknown_reason,
                )
            )
        return records

    def retained_datasets(self, min_bytes: int = 0) -> list[RetainedDataset]:
        """Audit inherited payloads, counting shared hard-linked objects once.

        Unknown logical sizes are always included, regardless of the threshold.
        This never opens an external target or reads a dataset value.
        """
        if (
            isinstance(min_bytes, bool)
            or not isinstance(min_bytes, int)
            or min_bytes < 0
        ):
            raise ValueError("min_bytes must be a nonnegative integer")
        return self._retained(self._snapshot(), min_bytes)

    def validate(
        self,
        source: Any = None,
        retained_warning_bytes: int | None = 100 * 1024**2,
    ) -> ValidationReport:
        """Inspect metadata only; provider execution/schema validation is writer-owned.

        The threshold is advisory. ``None`` disables large-inheritance warnings.
        Callbacks and result selectors are never invoked, even with a source.
        Whole dynamic bindings defer selected rank, bounds, and broadcasting
        checks to the writer; selection syntax and known dtypes are still checked.
        """
        if retained_warning_bytes is not None and (
            isinstance(retained_warning_bytes, bool)
            or not isinstance(retained_warning_bytes, int)
            or retained_warning_bytes < 0
        ):
            raise ValueError("retained_warning_bytes must be nonnegative or None")
        try:
            snapshot = self._snapshot()
        except (OSError, KeyError, TypeError, ValueError) as error:
            return ValidationReport(
                (
                    *self._diagnostics,
                    Diagnostic(
                        "error", "/", f"Cannot inspect template metadata: {error}"
                    ),
                )
            )
        return self._validate(snapshot, retained_warning_bytes)

    def _validate(
        self, snapshot: dict[str, NodePlan], threshold: int | None
    ) -> ValidationReport:
        diagnostics = list(self._diagnostics)

        def add(severity: str, path: str, message: str) -> None:
            diagnostics.append(Diagnostic(severity, path, message))

        paths: dict[Any, list[str]] = {}
        for path, plan in snapshot.items():
            paths.setdefault(plan.object_key, []).append(path)
            if plan.kind == "group":
                ancestor = posixpath.dirname(path) if path != "/" else ""
                while ancestor:
                    if snapshot[ancestor].object_key == plan.object_key:
                        add(
                            "error",
                            path,
                            "Cyclic hard-linked groups are not supported; "
                            "remove this alias",
                        )
                        break
                    ancestor = posixpath.dirname(ancestor) if ancestor != "/" else ""
            if plan.kind == "soft":
                target = posixpath.normpath(
                    posixpath.join(posixpath.dirname(path), plan.link.path)
                )
                if target not in snapshot:
                    add(
                        "warning",
                        path,
                        f"Unresolved soft link to {target}; inspect or "
                        "repair_links before writing",
                    )
            if plan.kind == "external":
                add(
                    "warning",
                    path,
                    f"External dependency {plan.link.filename}:{plan.link.path}; "
                    "output is not standalone (target not inspected)",
                )
                if not Path(plan.link.filename).is_absolute():
                    add(
                        "error",
                        path,
                        "Relative external links cannot be safely relocated; "
                        "replace with an explicitly resolved filename",
                    )

        for key, aliases in paths.items():
            node = self._objects[key]
            path = aliases[0]
            for subject, message in node.issues.items():
                location = (
                    path + "@" + subject.split(":", 1)[1]
                    if subject.startswith("attribute:")
                    else path
                )
                add("error", location, message)
            for adaptation in node.adaptations:
                add("warning", path, adaptation)
            if node.edits and len(aliases) > 1:
                add(
                    "warning",
                    path,
                    f"Edits affect all hard-link aliases: {_brief(aliases)}",
                )
            for i, (edit_path, subject, value) in enumerate(node.edits):
                for other_path, other_subject, other_value in node.edits[:i]:
                    if (
                        edit_path == other_path
                        or edit_path not in aliases
                        or other_path not in aliases
                        or subject != other_subject
                        or _equivalent(value, other_value)
                    ):
                        continue
                    if subject == "binding" and not _overlap(
                        value, other_value, node.creation.get("shape")
                    ):
                        continue
                    add(
                        "error",
                        path,
                        f"Conflicting alias edits through {other_path} and "
                        f"{edit_path} ({subject}); make the edits through "
                        "one canonical path",
                    )
            for name, binding in node.attributes.items():
                self._validate_binding(binding, path + "@" + name, add)
                if binding.kind == "literal" and _has_references(binding.value):
                    add(
                        "error",
                        path + "@" + name,
                        "Reference-bearing attributes are unsupported; "
                        "remove or replace this attribute explicitly",
                    )
                elif binding.kind == "literal":
                    try:
                        shape, dtype = _literal_spec(binding.value)
                        if _dtype_has(
                            dtype,
                            lambda d: (
                                d.kind == "O" and h5py.check_string_dtype(d) is None
                            ),
                        ):
                            raise ValueError(
                                "Unsupported object/variable-length literal attribute "
                                "dtype; use explicit text or numeric values"
                            )
                        if h5py.check_string_dtype(dtype) is not None:
                            self._validate_literal(
                                binding.value, shape, dtype, False, False
                            )
                    except (TypeError, ValueError) as error:
                        add("error", path + "@" + name, str(error))
                if name == "target" and binding.kind == "literal":
                    target = binding.value
                    if isinstance(target, np.ndarray) and target.size == 1:
                        target = target.flat[0]
                    if isinstance(target, bytes):
                        target = target.decode("utf-8", errors="replace")
                    if (
                        isinstance(target, str)
                        and target.startswith("/")
                        and target not in snapshot
                    ):
                        add(
                            "warning",
                            path + "@target",
                            f"Unresolved NeXus target path {_brief(target)}; "
                            "inspect or repair_links before writing",
                        )
            if node.kind != "dataset":
                continue
            retained = node.base_path is not None and not any(
                b.selection is None for b in node.bindings
            )
            if retained:
                if node.original_dtype.subdtype is not None:
                    add(
                        "error",
                        path,
                        "Retained top-level array datatype is unsupported; "
                        "remove or fully replace this dataset",
                    )
                for feature in node.features:
                    add(
                        "error",
                        path,
                        f"Retained {feature} is unsupported; "
                        "remove or fully replace this dataset",
                    )
                if _dtype_has(
                    node.original_dtype, lambda d: h5py.check_dtype(ref=d) is not None
                ):
                    add(
                        "error",
                        path,
                        "Retained object/region reference dtype is unsupported; "
                        "remove or fully replace this dataset",
                    )
                if "dtype" in node.explicit_creation and not _equivalent(
                    node.original_dtype, node.creation.get("dtype")
                ):
                    add(
                        "error",
                        path,
                        "Changing the dtype of a retained dataset requires "
                        "a whole replacement",
                    )
                if node.original_shape != node.creation.get("shape"):
                    add(
                        "error",
                        path,
                        "Changing the shape of a retained dataset requires "
                        "a whole replacement",
                    )
            self._validate_creation(node, path, add)
            dynamic_shape = False
            for binding in node.bindings:
                if binding.selection is None:
                    dynamic_shape = binding.kind in {"callback", "copy", "result"}
                self._validate_binding(binding, path, add)
                if binding.kind == "copy" and not isinstance(binding.value, str):
                    units = node.attributes.get("units")
                    declared = getattr(binding.value, "target_unit", None)
                    if (
                        declared is not None
                        and units is not None
                        and units.kind == "literal"
                        and not _equivalent(units.value, declared)
                    ):
                        add(
                            "warning",
                            path,
                            f"Declared conversion units {declared!r} conflict with "
                            f"destination units {_brief(units.value)!r}; "
                            "attributes do not convert data",
                        )
                try:
                    shape = None if dynamic_shape else node.creation.get("shape")
                    if binding.selection is not None:
                        if shape is None and not dynamic_shape:
                            raise ValueError(
                                "a selection update needs a known destination shape"
                            )
                        shape, _ = _selection(shape, binding.selection)
                    if binding.kind == "literal":
                        self._validate_literal(
                            binding.value,
                            shape,
                            node.creation.get("dtype"),
                            binding.selection is not None,
                            "dtype" in node.explicit_creation,
                        )
                except (TypeError, ValueError, OverflowError) as error:
                    add("error", path, str(error))

        for record in self._retained(snapshot):
            if record.logical_bytes is None:
                add(
                    "warning",
                    record.path,
                    f"Inherited logical size is unknown: {record.unknown_reason}",
                )
            elif threshold is not None and record.logical_bytes >= threshold:
                add(
                    "warning",
                    record.path,
                    f"Large inherited dataset: {record.logical_bytes} logical bytes; "
                    "inspect retained_datasets() before reusing experimental data",
                )
        return ValidationReport(tuple(dict.fromkeys(diagnostics)))

    @staticmethod
    def _validate_binding(binding: Binding, path: str, add: Any) -> None:
        for name in ("read_chunk_size", "write_chunk_size"):
            value = binding.options.get(name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
            ):
                add("error", path, f"{name} must be a positive integer number of rows")
        if binding.kind == "result":
            provider, selector = binding.value
            fields = getattr(provider, "result_fields", _RESULT_FIELDS)
            if isinstance(selector, str) and selector not in fields:
                add(
                    "error",
                    path,
                    f"Unknown result field {selector!r}; use a documented "
                    "field name or a labelled callback",
                )

    @staticmethod
    def _validate_creation(node: _Node, path: str, add: Any) -> None:
        spec = node.creation
        shape, dtype = spec.get("shape"), spec.get("dtype")
        dynamic_whole = any(
            b.selection is None and b.kind != "literal" for b in node.bindings
        )
        if shape is None and not dynamic_whole:
            add(
                "error",
                path,
                "Dataset needs a shape/value; null dataspaces are unsupported",
            )
        if dtype is None and not dynamic_whole:
            add("error", path, "Dataset needs a dtype or a literal value")
        if dtype is not None:
            if dtype.subdtype is not None:
                add(
                    "error",
                    path,
                    "Top-level array datatypes (subarray dtypes) are unsupported; "
                    "use a scalar-element output dtype",
                )
            reference = _dtype_has(dtype, lambda d: h5py.check_dtype(ref=d) is not None)
            if reference and (not dynamic_whole or "dtype" in node.explicit_creation):
                add("error", path, "Reference-bearing output dtype is unsupported")
            if _dtype_has(
                dtype,
                lambda d: (
                    d.kind == "O"
                    and h5py.check_string_dtype(d) is None
                    and h5py.check_dtype(ref=d) is None
                ),
            ):
                add(
                    "error",
                    path,
                    "Non-string object/variable-length output dtypes are unsupported",
                )
            if _dtype_has(dtype, lambda d: d.kind in "mMU"):
                add(
                    "error",
                    path,
                    "Unsupported HDF5 output dtype; "
                    "use an explicit numeric or string dtype",
                )
            if spec.get("fillvalue") is not None and dtype.subdtype is None:
                try:
                    NexusTemplate._validate_literal(
                        spec["fillvalue"], (), dtype, False, True
                    )
                except (TypeError, ValueError, OverflowError) as error:
                    add("error", path, f"Incompatible fillvalue: {error}")
        if spec.get("external") is not None:
            add("error", path, "External raw storage creation is unsupported")
        if shape is None:
            return
        maxshape, chunks = spec.get("maxshape"), spec.get("chunks")
        if maxshape is not None and (
            len(maxshape) != len(shape)
            or any(
                m is not None and m < s for s, m in zip(shape, maxshape, strict=True)
            )
        ):
            add(
                "error",
                path,
                "maxshape is incompatible with shape; "
                "supply a replacement specification",
            )
        if chunks is not None and chunks is not True:
            if (
                not isinstance(chunks, tuple)
                or len(chunks) != len(shape)
                or any(c <= 0 for c in chunks)
            ):
                add(
                    "error",
                    path,
                    "chunks must be True or positive dimensions matching "
                    "the dataset rank",
                )
            elif any(
                s
                and c > s
                and (
                    maxshape is None
                    or (
                        len(maxshape) == len(shape)
                        and maxshape[i] is not None
                        and c > maxshape[i]
                    )
                )
                for i, (s, c) in enumerate(zip(shape, chunks, strict=True))
            ):
                add(
                    "error",
                    path,
                    "Inherited chunks exceed the replacement shape; "
                    "specify compatible chunks/maxshape",
                )
        if not shape and any(
            spec.get(k) is not None
            for k in ("chunks", "compression", "shuffle", "fletcher32", "scaleoffset")
            if k not in {"shuffle", "fletcher32"} or spec.get(k)
        ):
            add(
                "error",
                path,
                "Scalar datasets cannot use chunking or filters; "
                "explicitly clear incompatible inherited properties",
            )
        if spec.get("compression_opts") is not None and spec.get("compression") is None:
            add("error", path, "compression_opts requires compression")
        if spec.get("scaleoffset") is not None and spec.get("fletcher32"):
            add("error", path, "scaleoffset and fletcher32 cannot be combined")

    @staticmethod
    def _validate_literal(
        value: Any,
        shape: tuple | None,
        dtype: np.dtype | None,
        selected: bool,
        explicit_dtype: bool,
    ) -> None:
        if isinstance(value, (h5py.Dataset, h5py.Group)):
            raise TypeError(
                "live HDF5 objects are not literals; use copy_from with a path"
            )
        array = np.asarray(value)
        if shape is not None:
            if selected:
                if np.broadcast_shapes(array.shape, shape) != shape:
                    raise ValueError(
                        f"literal shape {array.shape} cannot broadcast "
                        f"to selection shape {shape}"
                    )
            elif array.shape != shape:
                raise ValueError(
                    f"literal shape {array.shape} does not match dataset shape {shape}"
                )
        if dtype is None:
            return
        if _has_references(value):
            raise ValueError("Reference-bearing literal values are unsupported")
        string = h5py.check_string_dtype(dtype)
        if string is not None:
            for item in array.flat:
                if not isinstance(item, (str, bytes, np.str_, np.bytes_)):
                    raise ValueError(
                        "string destinations require string/bytes literal values"
                    )
                encoded = (
                    item.encode(string.encoding)
                    if isinstance(item, str)
                    else bytes(item)
                )
                if string.length is None and b"\0" in encoded:
                    raise ValueError(
                        "variable-length strings cannot contain embedded NUL"
                    )
                if string.length is not None and len(encoded) > string.length:
                    raise ValueError(
                        "literal would truncate a fixed-length string; "
                        "use a whole replacement with a wider dtype"
                    )
            return
        if array.dtype.kind == "O":
            raise ValueError(
                "object literals require explicit text or numeric array values"
            )
        if array.dtype.kind == "c" and dtype.kind != "c":
            raise ValueError(
                "complex literals require a complex dtype "
                "or explicitly real-valued input"
            )
        if dtype.kind in "iu" and array.size:
            if array.dtype.kind not in "biuf":
                raise ValueError("integer destinations require real numeric literals")
            if array.dtype.kind == "f" and (
                not np.isfinite(array).all()
                or not np.equal(array, np.floor(array)).all()
            ):
                raise ValueError(
                    "non-integral or non-finite literal cannot be assigned "
                    "to an integer dtype"
                )
            bounds = np.iinfo(dtype)
            if array.min().item() < bounds.min or array.max().item() > bounds.max:
                raise ValueError(f"literal overflows destination dtype {dtype}")
        elif not np.can_cast(array.dtype, dtype, casting="safe") and not explicit_dtype:
            raise ValueError(
                f"unsafe implicit cast from {array.dtype} to {dtype}; "
                "specify dtype explicitly"
            )

    def summary(self, source: Any = None, limit: int = 20) -> str:
        """Bounded help from metadata, never from callback execution or sampling."""
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("limit must be a positive integer")
        try:
            snapshot = self._snapshot()
        except (OSError, KeyError, TypeError, ValueError) as error:
            return f"{self!r}\nerror /: {_brief(str(error))}"
        labels: dict[int, tuple[str, Any]] = {}
        counters = {"sample": 0, "source": 0}

        def provider_label(value: Any) -> str:
            provider = getattr(
                value, "execution_source", getattr(value, "source", value)
            )
            key = id(provider)
            if key not in labels:
                sampled = any(
                    getattr(provider, name, None) is not None
                    for name in ("sampling", "sampling_settings", "settings")
                )
                prefix = "sample" if sampled else "source"
                counters[prefix] += 1
                labels[key] = (f"{prefix}{counters[prefix]}", provider)
            return labels[key][0]

        def describe(binding: Binding) -> str:
            description = binding.options.get("description")
            if binding.kind == "literal":
                text = "literal " + _brief(binding.value)
            elif binding.kind == "callback":
                text = _brief(binding.value)
            elif binding.kind == "copy":
                if isinstance(binding.value, str):
                    text = "copy source " + binding.value
                else:
                    text = (
                        provider_label(binding.value)
                        + " -> "
                        + self._provider_description(binding.value)
                    )
            else:
                provider, selector = binding.value
                text = provider_label(provider) + " result -> " + _brief(selector)
            if description:
                text += " (user description: " + _brief(description) + ")"
            if binding.selection is not None:
                text += " selection=" + _brief(binding.selection)
            if binding.options.get("columns") is not None:
                text += " columns=" + _brief(binding.options["columns"])
            return _brief(text, 220)

        # Register every provider before limiting rows so shared labels stay stable.
        for plan in snapshot.values():
            for binding in (*plan.bindings, *plan.attributes.values()):
                if binding.kind == "copy" and not isinstance(binding.value, str):
                    provider_label(binding.value)
                elif binding.kind == "result":
                    provider_label(binding.value[0])
        base = _brief(str(self.base_path)) if self.base_path else "empty"
        lines = [f"NexusTemplate: base={base}"]
        for label, provider in islice(labels.values(), limit):
            lines.append(f"{label}: {self._provider_description(provider)}")
            stages = getattr(provider, "preprocessors", ())
            if stages:
                lines.append(
                    "  raw read dependencies: "
                    + _brief(provider.required_variables)
                    + "; raw dtypes and accepted rows unknown until execution"
                )
                for index, stage in enumerate(islice(stages, limit), 1):
                    lines.append(
                        f"  preprocessing {index}: " + _brief(stage.description, 220)
                    )
                    lines.append("    declared output schema: " + _brief(stage.schema))
                if len(stages) > limit:
                    lines.append(
                        f"  ... {len(stages) - limit} more preprocessing stages"
                    )
        if len(labels) > limit:
            lines.append(f"... {len(labels) - limit} more shared sources")
        lines.append(
            f"Nodes: {len(snapshot)} (attributes do not imply data transformations)"
        )
        for path, plan in islice(snapshot.items(), limit):
            inherited = plan.base_path is not None and not any(
                b.selection is None for b in plan.bindings
            )
            origin = f"inherited from {plan.base_path}" if inherited else "new/replaced"
            spec = "; ".join(
                f"{k}={_brief(v, 60)}" for k, v in islice(plan.creation.items(), 5)
            )
            lines.append(_brief(f"{path}: {plan.kind}; {origin}; {spec}", 240))
            node = self._objects[plan.object_key]
            if node.description:
                lines.append("  description: " + _brief(node.description))
            lines.extend(
                "  " + describe(binding) for binding in islice(plan.bindings, 3)
            )
            for binding in islice(plan.bindings, 3):
                stages = getattr(binding.value, "chunk_operations", ())
                for index, stage in enumerate(islice(stages, 3), 1):
                    lines.append(
                        f"    chunk transform {index}: inputs={_brief(stage.inputs)}; "
                        + _brief(stage.description, 180)
                    )
                    lines.append(
                        "      declared output schema: " + _brief(stage.schema)
                    )
                if len(stages) > 3:
                    lines.append(f"    ... {len(stages) - 3} more chunk transforms")
            if len(plan.bindings) > 3:
                lines.append(f"  ... {len(plan.bindings) - 3} more bindings")
            for name, binding in islice(plan.attributes.items(), 3):
                lines.append("  @" + _brief(name, 40) + ": " + describe(binding))
            if len(plan.attributes) > 3:
                lines.append(f"  ... {len(plan.attributes) - 3} more attributes")
            if plan.link is not None:
                lines.append("  link: " + _brief(plan.link.path))
        if len(snapshot) > limit:
            lines.append(f"... {len(snapshot) - limit} more nodes")
        retained = self._retained(snapshot)
        known = sum(r.logical_bytes for r in retained if r.logical_bytes is not None)
        unknown = [r for r in retained if r.logical_bytes is None]
        lines.append(
            f"Retained: {len(retained)} unique datasets; {known} known logical bytes; "
            f"{len(unknown)} unknown sizes"
        )
        for record in sorted(retained, key=lambda r: -(r.logical_bytes or 0))[:limit]:
            logical = (
                record.logical_bytes if record.logical_bytes is not None else "unknown"
            )
            lines.append(
                _brief(
                    f"  {record.path}: logical={logical}, "
                    f"allocated={record.storage_bytes}; "
                    f"{'partial' if record.partial else 'full'}; "
                    f"aliases={_brief(record.aliases)}",
                    240,
                )
            )
        lines.extend(
            _brief(f"  unknown size: {record.path}: {record.unknown_reason}", 240)
            for record in unknown[:limit]
        )
        lines.extend("Removed: " + _brief(path) for path in self._removals[:limit])
        for old, new in self._moves[:limit]:
            lines.append(_brief(f"Moved: {old} -> {new}", 240))
        for old, new in self._repairs[:limit]:
            lines.append(_brief(f"Link repair: {old} -> {new}", 240))
        report = self._validate(snapshot, 100 * 1024**2)
        lines.extend(
            _brief(
                f"{diagnostic.severity} {diagnostic.path}: {diagnostic.message}",
                240,
            )
            for diagnostic in report.diagnostics[:limit]
        )
        if len(report.diagnostics) > limit:
            lines.append(f"... {len(report.diagnostics) - limit} more diagnostics")
        return "\n".join(lines)

    @staticmethod
    def _provider_description(provider: Any) -> str:
        variable = getattr(provider, "variable", None)
        operations = getattr(provider, "operations", None)
        if variable is not None and isinstance(operations, (tuple, list)):
            steps = [_brief(variable, 40)]
            for operation in islice(operations, 4):
                description = getattr(operation, "description", operation)
                steps.append(_brief(description, 110))
            if len(operations) > 4:
                steps.append(f"... ({len(operations) - 4} more operations)")
            return _brief(" -> ".join(steps), 210)
        description = getattr(provider, "description", None)
        text = _brief(
            description
            if description is not None and not callable(description)
            else provider
        )
        settings = getattr(provider, "sampling", None)
        if settings is not None and hasattr(settings, "n_samples"):
            text = (
                f"weighted sample requested={settings.n_samples}, "
                f"seed={getattr(settings, 'seed', None)}; " + text
            )
        return _brief(text, 220)

    def __repr__(self) -> str:
        base = _brief(str(self.base_path)) if self.base_path else None
        return (
            f"NexusTemplate(base_path={base!r}, moves={len(self._moves)}, "
            f"removals={len(self._removals)})"
        )


def create_template(path: str | Path) -> NexusTemplate:
    """Record a base path without opening it or copying any content."""
    return NexusTemplate(path)


def create_empty_template() -> NexusTemplate:
    """Create an empty root with no file, temporary file, or assumed NX layout."""
    return NexusTemplate()
