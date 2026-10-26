# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 McCode-dev contributors (https://github.com/mccode-dev)
"""Execute a NeXus layout recipe without changing either input file.

Ordinary HDF5 transfers are tiled. Event input uses the existing row iterator,
and sampled input uses the existing collecting sampler once per source identity.
Only writer-side transformations and projections are output-batched; this does
not remove the sampler's O(n_samples * variables) memory requirement.
"""

from __future__ import annotations

import os
import sys
import tempfile
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .NexusSource import EventColumn, EventSource, SourceContext, _checked_cast
from .NexusTemplate import (
    Binding,
    Diagnostic,
    NexusTemplate,
    NodePlan,
    RetainedDataset,
    _dtype_has,
    _has_references,
    _literal_spec,
    _selection,
)
from .ReadNeXus import _validate_chunk_size
from .Sampling import sample_event_chunks


@dataclass(frozen=True)
class SourceResult:
    """Statistics for one shared source execution, not one destination column."""

    n_written: int
    n_selected: int | None = None
    total_weight: float | None = None
    total_weight_variance: float | None = None
    effective_duration: float | None = None
    effective_duration_variance: float | None = None


@dataclass(frozen=True)
class DestinationResult:
    shape: tuple[int, ...]
    dtype: np.dtype
    n_written: int
    source: EventSource | str | None = None
    transformations: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProgressUpdate:
    """Coarse execution progress; rows and totals may be unavailable."""

    phase: str
    path: str | None = None
    rows: int = 0
    total: int | None = None


@dataclass(repr=False)
class WriteReport:
    output: Path
    sources: dict[EventSource, SourceResult] = field(default_factory=dict)
    destinations: dict[str, DestinationResult] = field(default_factory=dict)
    diagnostics: tuple[Diagnostic, ...] = ()
    retained_datasets: tuple[RetainedDataset, ...] = ()
    link_repairs: tuple[tuple[str, str], ...] = ()

    def __repr__(self) -> str:
        return (
            f"WriteReport(output={str(self.output)!r}, sources={len(self.sources)}, "
            f"datasets={len(self.destinations)}, diagnostics={len(self.diagnostics)})"
        )


def _tiles(shape: tuple[int, ...], limit: int) -> Iterator[tuple[slice, ...]]:
    """Bound ordinary-copy slabs by elements, including very wide single rows."""
    if not shape:
        yield ()
        return
    if 0 in shape:
        return
    tile = [1] * len(shape)
    remaining = limit
    for axis in reversed(range(len(shape))):
        tile[axis] = min(shape[axis], remaining)
        remaining = max(1, remaining // tile[axis])
    start = [0] * len(shape)
    while True:
        yield tuple(
            slice(offset, min(offset + step, size))
            for offset, size, step in zip(start, shape, tile, strict=True)
        )
        for axis in reversed(range(len(shape))):
            start[axis] += tile[axis]
            if start[axis] < shape[axis]:
                break
            start[axis] = 0
        else:
            return


def _supported_dtype(dtype: np.dtype) -> None:
    if dtype.subdtype is not None:
        raise TypeError("Top-level HDF5 array datatypes are not supported")
    if _dtype_has(dtype, lambda d: h5py.check_dtype(ref=d) is not None):
        raise TypeError("Object/region reference datasets are not supported")
    if _dtype_has(
        dtype, lambda d: d.kind == "O" and h5py.check_string_dtype(d) is None
    ):
        raise TypeError("Non-string variable-length/object datasets are not supported")
    if _dtype_has(dtype, lambda d: d.kind in "mMU"):
        raise TypeError(f"Unsupported HDF5 dtype {dtype}")


def _cast_value(value: Any, dtype: np.dtype, *, explicit: bool) -> np.ndarray:
    if value is None:
        raise ValueError("Binding value is unavailable (None); supply a fallback")
    if isinstance(value, (h5py.Dataset, h5py.Group)):
        raise TypeError("Callbacks must return values, not live HDF5 objects")
    if _has_references(value):
        raise TypeError("Reference-bearing values are not supported")
    dtype = np.dtype(dtype)
    _supported_dtype(dtype)
    array = np.asarray(value)
    string = h5py.check_string_dtype(dtype)
    if string is not None:
        encoded = []
        for item in array.flat:
            if not isinstance(item, (str, bytes, np.str_, np.bytes_)):
                raise TypeError("String destinations require text or bytes")
            raw = item.encode(string.encoding) if isinstance(item, str) else bytes(item)
            if string.length is not None and len(raw) > string.length:
                raise ValueError(
                    "Value would truncate a fixed-length string; use a whole "
                    "replacement with a wider dtype"
                )
            if string.length is None and b"\0" in raw:
                raise ValueError("Variable-length strings cannot contain NUL")
            encoded.append(raw)
        if string.length is not None:
            return np.asarray(encoded, dtype=dtype).reshape(array.shape)
        return np.asarray(encoded, dtype=object).reshape(array.shape)
    if dtype.kind in "iufc":
        if (
            not explicit
            and dtype.kind in "fc"
            and not np.can_cast(array.dtype, dtype, casting="safe")
        ):
            raise ValueError(
                f"Unsafe implicit cast from {array.dtype} to {dtype}; specify dtype"
            )
        return _checked_cast(
            array.astype(np.uint8) if array.dtype.kind == "b" else array, dtype
        )
    if dtype.kind == "b":
        if array.dtype.kind not in "biuf" or not np.all((array == 0) | (array == 1)):
            raise ValueError("Boolean destinations require boolean or 0/1 values")
    elif not np.can_cast(array.dtype, dtype, casting="safe"):
        raise ValueError(f"Unsafe cast from {array.dtype} to {dtype}")
    return array.astype(dtype, copy=False)


def _output_dtype(
    plan: NodePlan, incoming: np.dtype, explicit_keys: set[str], *, whole: bool
) -> np.dtype:
    dtype = np.dtype(plan.creation.get("dtype", incoming))
    # A whole, implicit text replacement may grow rather than silently truncate.
    if whole and "dtype" not in explicit_keys:
        old, new = h5py.check_string_dtype(dtype), h5py.check_string_dtype(incoming)
        if old is not None and new is not None:
            if old.encoding == "utf-8":
                length = (
                    None
                    if old.length is None or new.length is None
                    else max(old.length, new.length)
                )
                dtype = np.dtype(h5py.string_dtype("utf-8", length=length))
            elif new.length is None or (
                old.length is not None and new.length > old.length
            ):
                dtype = incoming
    _supported_dtype(dtype)
    return dtype


@contextmanager
def _destination_error(path: str, source: Any = None) -> Iterator[None]:
    try:
        yield
    except (ValueError, TypeError, OverflowError, KeyError, OSError) as error:
        origin = (
            source if isinstance(source, str) else getattr(source, "description", "")
        )
        message = f"Destination {path}"
        if origin:
            message += f" from {origin}"
        raise type(error)(f"{message}: {error}") from error


def _selected_shape(shape: tuple[int, ...], selection: Any) -> tuple[int, ...]:
    return shape if selection is None else _selection(shape, selection)[0]


def _assign(
    dataset: h5py.Dataset, value: Any, selection: Any, *, explicit: bool
) -> None:
    cast = _cast_value(value, dataset.dtype, explicit=explicit)
    wanted = _selected_shape(dataset.shape, selection)
    if selection is None:
        if cast.shape != wanted:
            raise ValueError(f"Value shape {cast.shape} differs from dataset {wanted}")
    elif np.broadcast_shapes(cast.shape, wanted) != wanted:
        raise ValueError(f"Value shape {cast.shape} cannot broadcast to {wanted}")
    dataset[() if selection is None else selection] = cast


def _target_slice(
    destination: h5py.Dataset, selection: Any, slab: tuple[slice, ...]
) -> tuple:
    if selection is None:
        return slab
    _, regions = _selection(destination.shape, selection)
    items = selection if isinstance(selection, tuple) else (selection,)
    missing = destination.ndim - sum(item is not Ellipsis for item in items)
    expanded = []
    for item in items:
        expanded.extend([slice(None)] * missing if item is Ellipsis else [item])
    expanded.extend([slice(None)] * (destination.ndim - len(expanded)))
    target, axis = [], 0
    for item, region in zip(expanded, regions, strict=True):
        if isinstance(item, (int, np.integer)):
            target.append(region.start)
        else:
            part = slab[axis]
            target.append(
                slice(
                    region.start + part.start * region.step,
                    region.start + part.stop * region.step,
                    region.step,
                )
            )
            axis += 1
    return tuple(target)


def _copy_dataset(
    source: h5py.Dataset,
    destination: h5py.Dataset,
    read_size: int,
    write_size: int,
    *,
    selection: Any = None,
    explicit: bool = False,
) -> None:
    if source.shape != _selected_shape(destination.shape, selection):
        raise ValueError(
            f"Source {source.name} shape {source.shape} "
            "differs from destination selection"
        )
    for slab in _tiles(source.shape, read_size):
        block = source[slab]
        block = _cast_value(block, destination.dtype, explicit=explicit)
        for part in _tiles(block.shape, write_size):
            absolute = tuple(
                slice(outer.start + inner.start, outer.start + inner.stop)
                for outer, inner in zip(slab, part, strict=True)
            )
            destination[_target_slice(destination, selection, absolute)] = block[part]


def _same_file(left: Path, right: Path) -> bool:
    if left.resolve() == right.resolve():
        return True
    return left.exists() and right.exists() and os.path.samefile(left, right)


def write_nexus(
    template: NexusTemplate,
    output: str | os.PathLike[str],
    *,
    source: Any = None,
    entry_number: int | None = None,
    read_chunk_size: int = 100_000,
    write_chunk_size: int = 100_000,
    overwrite: bool = False,
    validators: Any = (),
    progress: Callable[[ProgressUpdate], None] | None = None,
) -> WriteReport:
    """Write a recipe atomically; existing outputs require explicit permission.

    Ordinary HDF5 copies bound each tiled transfer by the batch size in elements
    (rows for one-dimensional data). Event batch sizes count aligned rows. Physical
    HDF5 ``chunks`` are independent dataset creation properties. Inputs must remain
    unchanged during the write. Borrowed Read/SourceContext handles remain open.

    Validators receive the completed, open temporary h5py.File before publication.
    Progress callbacks run before publication too, so their failures are atomic.
    Warnings are available in the report's diagnostics, not Python warnings.
    """
    if not isinstance(template, NexusTemplate):
        raise TypeError("template must be a NexusTemplate")
    _validate_chunk_size(read_chunk_size, allow_none=False)
    _validate_chunk_size(write_chunk_size, allow_none=False)
    if not isinstance(overwrite, bool):
        raise TypeError("overwrite must be a bool")
    if progress is not None and not callable(progress):
        raise TypeError("progress must be callable or None")
    validators = (validators,) if callable(validators) else tuple(validators)
    if any(not callable(validator) for validator in validators):
        raise TypeError("validators must contain callables")
    output = Path(output).expanduser().absolute()
    if not output.parent.is_dir():
        raise FileNotFoundError(f"Output directory does not exist: {output.parent}")
    if output.is_dir():
        raise IsADirectoryError(str(output))
    if os.path.lexists(output) and not overwrite:
        raise FileExistsError(str(output))

    def notify(phase: str, path: str | None = None, rows: int = 0) -> None:
        if progress is not None:
            progress(ProgressUpdate(phase, path, rows))

    temporary: Path | None = None
    with ExitStack() as inputs:
        context = inputs.enter_context(SourceContext(source, entry_number=entry_number))
        input_paths = [template.base_path] if template.base_path is not None else []
        if context._data is not None:
            input_paths.append(Path(context.raw.filename))
        if any(_same_file(output, path) for path in input_paths):
            raise ValueError("Output path aliases a source or base-template input file")
        base = (
            inputs.enter_context(h5py.File(template.base_path, "r"))
            if template.base_path is not None
            else None
        )
        notify("preflight")
        validation = template.validate(source=context)
        validation.raise_if_errors()
        snapshot = template._snapshot()
        report = WriteReport(
            output,
            diagnostics=validation.diagnostics,
            retained_datasets=tuple(template.retained_datasets()),
            link_repairs=tuple(template._repairs),
        )
        unique: dict[Any, NodePlan] = {}
        aliases: dict[Any, list[str]] = {}
        for path, plan in snapshot.items():
            unique.setdefault(plan.object_key, plan)
            aliases.setdefault(plan.object_key, []).append(path)
        schemas: dict[EventSource, dict[str, np.dtype]] = {}
        consumers: dict[EventSource, list[NodePlan]] = {}
        event_keys = set()

        def inspect_dataset(dataset: h5py.Dataset) -> None:
            if _same_file(output, Path(dataset.file.filename)):
                raise ValueError("Output path aliases a resolved source input file")
            if dataset.shape is None:
                raise TypeError("Null source datasets are unsupported")
            if dataset.is_virtual or dataset.external:
                raise TypeError(
                    "Source virtual datasets and external raw storage are unsupported"
                )
            _supported_dtype(dataset.dtype)

        def register(provider: Any) -> EventSource:
            if not isinstance(provider, EventSource):
                raise TypeError("Result bindings require an EventSource")
            if provider not in schemas:
                reader, components, schema, _ = provider._inspect(context)
                for component in components:
                    inspect_dataset(reader.get_info_entry(component)["events"])
                schemas[provider] = schema
                consumers[provider] = []
            return provider

        def full_binding(plan: NodePlan) -> Binding | None:
            return next((b for b in plan.bindings if b.selection is None), None)

        for plan in unique.values():
            if plan.kind == "external" and _same_file(output, Path(plan.link.filename)):
                raise ValueError("Output path aliases a retained external input file")
            for binding in (*plan.bindings, *plan.attributes.values()):
                with _destination_error(plan.path, binding.value):
                    if binding.kind == "result":
                        register(binding.value[0])
                    elif binding.kind == "copy":
                        value = binding.value
                        if isinstance(value, (EventSource, EventColumn)):
                            if binding.selection is not None:
                                raise ValueError(
                                    "Event projections require whole-dataset bindings"
                                )
                            provider = register(
                                value.source
                                if isinstance(value, EventColumn)
                                else value
                            )
                            if isinstance(value, EventColumn):
                                value.output_dtype(schemas[provider][value.variable])
                            else:
                                columns = binding.options.get("columns")
                                if not columns or len(set(columns)) != len(columns):
                                    raise ValueError("Event tables need unique columns")
                                for column in columns:
                                    if column not in provider.variables:
                                        raise ValueError(
                                            f"Undeclared table column {column}"
                                        )
                            consumers[provider].append(plan)
                            event_keys.add(plan.object_key)
                        elif isinstance(value, str):
                            dataset = context._dataset(value)
                            inspect_dataset(dataset)
                        else:
                            raise TypeError("Unsupported copy provider")

        fd, name = tempfile.mkstemp(
            prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
        )
        os.close(fd)
        temporary = Path(name)
        published = False
        try:
            with h5py.File(temporary, "w") as destination:
                created: dict[Any, h5py.Group | h5py.Dataset] = {
                    snapshot["/"].object_key: destination
                }

                def attach(
                    plan: NodePlan, obj: Any, paths: tuple[str, ...] | None = None
                ) -> None:
                    for path in aliases[plan.object_key] if paths is None else paths:
                        if path == "/":
                            continue
                        if path in destination:
                            if destination[path].id != obj.id:
                                raise ValueError(f"Conflicting output alias {path}")
                        else:
                            destination[path] = obj

                for plan in snapshot.values():
                    if plan.kind == "group" and plan.path != "/":
                        if plan.object_key not in created:
                            created[plan.object_key] = destination.create_group(
                                plan.path
                            )
                        # Parent groups must exist before their aliases are attached.
                        attach(plan, created[plan.object_key], (plan.path,))

                def explicit_keys(plan: NodePlan) -> set[str]:
                    return template._objects[plan.object_key].explicit_creation

                def create(
                    plan: NodePlan, shape: tuple, dtype: np.dtype, *, grow: bool = False
                ) -> h5py.Dataset:
                    spec = dict(plan.creation)
                    spec.update(shape=shape, dtype=dtype)
                    if grow and "maxshape" not in spec:
                        spec["maxshape"] = (None, *shape[1:])
                    with _destination_error(plan.path):
                        obj = destination.create_dataset(plan.path, **spec)
                        created[plan.object_key] = obj
                        attach(plan, obj)
                    return obj

                def record(plan: NodePlan, origin: Any = None) -> None:
                    obj = created[plan.object_key]
                    provider = (
                        origin.source if isinstance(origin, EventColumn) else origin
                    )
                    operations = (
                        tuple(op.description for op in origin.operations)
                        if isinstance(origin, EventColumn)
                        else ()
                    )
                    result = DestinationResult(
                        obj.shape,
                        obj.dtype,
                        obj.shape[0] if obj.shape else 1,
                        provider,
                        operations,
                    )
                    for path in aliases[plan.object_key]:
                        report.destinations[path] = result

                def resolve(binding: Binding) -> Any:
                    if binding.kind == "literal":
                        return binding.value
                    if binding.kind == "callback":
                        return binding.value(context)
                    if binding.kind == "result":
                        provider, selector = binding.value
                        result = report.sources[provider]
                        value = (
                            getattr(result, selector)
                            if isinstance(selector, str)
                            else selector(result)
                        )
                        if value is None:
                            raise ValueError(
                                f"Result {selector!r} is unavailable (None); "
                                "supply a fallback callback"
                            )
                        return value
                    raise TypeError("This binding is not a small value")

                def apply_update(plan: NodePlan, binding: Binding) -> None:
                    obj = created[plan.object_key]
                    with _destination_error(plan.path, binding.value):
                        if binding.kind == "copy":
                            _copy_dataset(
                                context._dataset(binding.value),
                                obj,
                                binding.options.get("read_chunk_size", read_chunk_size),
                                binding.options.get(
                                    "write_chunk_size", write_chunk_size
                                ),
                                selection=binding.selection,
                                explicit="dtype" in explicit_keys(plan),
                            )
                        else:
                            _assign(
                                obj,
                                resolve(binding),
                                binding.selection,
                                explicit="dtype" in explicit_keys(plan),
                            )

                def ordinary(plan: NodePlan) -> None:
                    binding = full_binding(plan)
                    with _destination_error(
                        plan.path, binding.value if binding else None
                    ):
                        value = None
                        original = None
                        if binding is not None and binding.kind == "copy":
                            original = context._dataset(binding.value)
                            shape, incoming = original.shape, original.dtype
                        elif binding is not None:
                            value = resolve(binding)
                            if value is None:
                                raise ValueError("Binding value is unavailable (None)")
                            shape, incoming = _literal_spec(value)
                        elif plan.base_path is not None:
                            original = base[plan.base_path]
                            shape, incoming = original.shape, original.dtype
                        else:
                            shape = plan.creation["shape"]
                            incoming = plan.creation["dtype"]
                        dtype = _output_dtype(
                            plan,
                            np.dtype(incoming),
                            explicit_keys(plan),
                            whole=binding is not None,
                        )
                        obj = create(plan, shape, dtype)
                        if original is not None:
                            options = binding.options if binding is not None else {}
                            _copy_dataset(
                                original,
                                obj,
                                options.get("read_chunk_size", read_chunk_size),
                                options.get("write_chunk_size", write_chunk_size),
                                explicit="dtype" in explicit_keys(plan),
                            )
                        elif binding is not None:
                            _assign(
                                obj,
                                value,
                                None,
                                explicit="dtype" in explicit_keys(plan),
                            )
                        record(
                            plan,
                            binding.value
                            if binding and binding.kind == "copy"
                            else None,
                        )
                        notify("copy", plan.path, obj.shape[0] if obj.shape else 1)

                for plan in unique.values():
                    if plan.kind != "dataset" or plan.object_key in event_keys:
                        continue
                    binding = full_binding(plan)
                    if binding is None or binding.kind != "result":
                        ordinary(plan)

                def execute_provider(
                    provider: EventSource, plans: list[NodePlan]
                ) -> None:
                    # A local execution scope releases every payload view before the
                    # next independent collecting sampler is called.
                    schema = dict(schemas[provider])
                    sampled = None
                    if provider.sampling is not None:
                        notify("sample", rows=0)
                        sampled = sample_event_chunks(
                            provider.iter_chunks(context), provider.sampling
                        )
                        count = len(next(iter(sampled.values()))) if sampled else 0
                        report.sources[provider] = SourceResult(
                            count,
                            count,
                            sampled.total_weight,
                            sampled.total_weight_variance,
                            sampled.effective_duration,
                            sampled.effective_duration_variance,
                        )
                        schema[provider.sampling.weight_key] = np.dtype("float64")
                        batches = (sampled,) if count else ()
                    else:
                        count = provider.row_count(context)
                        batches = provider.iter_chunks(context)
                    batch_size = min(
                        [write_chunk_size]
                        + [
                            full_binding(plan).options.get(
                                "write_chunk_size", write_chunk_size
                            )
                            for plan in plans
                        ]
                    )
                    for plan in plans:
                        binding = full_binding(plan)
                        column = binding.value
                        if isinstance(column, EventColumn):
                            incoming = column.output_dtype(schema[column.variable])
                            trailing = ()
                        else:
                            names = binding.options["columns"]
                            incoming = np.result_type(*(schema[name] for name in names))
                            trailing = (len(names),)
                        dtype = _output_dtype(
                            plan, incoming, explicit_keys(plan), whole=True
                        )
                        create(
                            plan,
                            (0 if count is None else count, *trailing),
                            dtype,
                            grow=count is None,
                        )
                    offset = 0
                    for chunk in batches:
                        length = len(next(iter(chunk.values())))
                        for start in range(0, length, batch_size):
                            stop = min(start + batch_size, length)
                            batch = {
                                key: values[start:stop] for key, values in chunk.items()
                            }
                            for plan in plans:
                                binding = full_binding(plan)
                                obj = created[plan.object_key]
                                with _destination_error(plan.path, binding.value):
                                    if count is None:
                                        obj.resize(offset + stop - start, axis=0)
                                    if isinstance(binding.value, EventColumn):
                                        values = binding.value.apply(batch)
                                    else:
                                        values = np.column_stack(
                                            [
                                                batch[name]
                                                for name in binding.options["columns"]
                                            ]
                                        )
                                    _assign(
                                        obj,
                                        values,
                                        slice(offset, offset + stop - start),
                                        explicit="dtype" in explicit_keys(plan),
                                    )
                            offset += stop - start
                            notify("write", rows=offset)
                    if count is not None and offset != count:
                        raise ValueError(
                            "Event source row count changed during writing"
                        )
                    if provider.sampling is None:
                        report.sources[provider] = SourceResult(offset)
                    for plan in plans:
                        record(plan, full_binding(plan).value)

                for provider, plans in consumers.items():
                    execute_provider(provider, plans)

                for plan in unique.values():
                    if plan.kind == "dataset":
                        binding = full_binding(plan)
                        if binding is not None and binding.kind == "result":
                            ordinary(plan)
                        for update in plan.bindings:
                            if update.selection is not None:
                                apply_update(plan, update)
                    if plan.kind not in {"group", "dataset"}:
                        continue
                    obj = created[plan.object_key]
                    for name, binding in plan.attributes.items():
                        with _destination_error(plan.path + "@" + name):
                            value = resolve(binding)
                            _, dtype = _literal_spec(value)
                            cast = _cast_value(value, dtype, explicit=True)
                            obj.attrs.create(name, cast, dtype=dtype)
                for plan in snapshot.values():
                    if plan.kind in {"soft", "external"} and not destination.get(
                        plan.path, getlink=True
                    ):
                        destination[plan.path] = plan.link
                notify("validate")
                for validator in validators:
                    validator(destination)
                destination.flush()
            notify("finalize")
            if overwrite:
                os.replace(temporary, output)
            else:
                # Unlike rename on POSIX, link fails atomically if output appeared.
                os.link(temporary, output)
            published = True
            return report
        finally:
            failure = sys.exc_info()[1]
            try:
                temporary.unlink(missing_ok=True)
            except OSError as error:
                message = f"Temporary file cleanup failed for {temporary}: {error}"
                if published:
                    report.diagnostics += (
                        Diagnostic("warning", str(temporary), message),
                    )
                elif failure is not None:
                    if hasattr(failure, "add_note"):
                        failure.add_note(message)
                else:
                    raise
