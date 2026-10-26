# Custom NeXus Files

Use `mcstastox.create_template`, `create_empty_template`, `event_source`, and
`write_nexus` for custom HDF5/NeXus layouts with just NumPy and h5py, without Scipp.
The package also exports `NexusTemplate`, `SourceContext`, `EventSource`,
`EventColumn`, and `WriteReport`.

For an executable introduction with a real McStas simulation, see
[Write Sampled Events to a Custom NeXus File](custom_nexus_demo.ipynb). It uses
an empty template first, then retains McStas metadata and geometry using the
source file as its reference. Both examples write sampled pixel IDs and
explicitly pulse-relative times; the second replaces only the shared raw event
table and preserves its group aliases.

A `NexusTemplate` records instructions, not a clone or dataset payloads. Writing
combines retained original-template content, an optional McStas source, and your
instructions without modifying either input. Keep inputs complete and unchanged
through inspection/writing; SWMR reading is not a snapshot.

## Choose a Layout

Start from a facility skeleton or an inspected experimental file:

```python
import mcstastox as mx

template = mx.create_template("instrument-template.nxs")
template["/entry/title"].set_value("Simulated events")
template["/entry/end_time"].set_value("2026-10-22T10:07:28Z", selection=0)
```

Assume a scalar text `title` and length-one text `end_time` wide enough for the timestamp.
Unchanged content, including old events/logs, is retained unless explicitly removed.
Construction is lazy; inspection uses read-only handles, not persistent h5py objects.

`create_empty_template()` needs no base file or temporary skeleton. Specify the layout
and NeXus attributes explicitly, using the same editing/writing API. Add parents before
children. Unbound zero-length datasets remain empty; whole bindings replace them.

## One Shared Sample

Illustrative input `simulation/mccode.h5` must have `entry2`, component `Detector`,
exact variables `id`, `t`, `p`, and only IDs 0, 1, 2. Assume `t` in seconds and finite,
nonnegative count-rate weights `p` with a positive sum. Adapt these to your instrument.

```python
import mcstastox as mx

template = mx.create_empty_template()
entry = template.add_group("/entry")
entry.attrs["NX_class"] = "NXentry"
entry.add_dataset("title", value="McStas events in a custom layout")
output_events = entry.add_group("events")
output_events.attrs["NX_class"] = "NXevent_data"
output_events.add_dataset("event_id", shape=(0,), dtype="int64")
output_events.add_dataset("event_time_offset", shape=(0,), dtype="float64")
entry.add_dataset("duration", shape=(), dtype="float64")

events = mx.event_source(
    component_name="Detector", variables=("id", "t"), read_chunk_size=100_000,
).sample(mx.SamplingSettings(n_samples=100_000, seed=42))
template["/entry/events/event_id"].copy_from(
    events["id"].map_values({0: 1001, 1: 1002, 2: 1010}, dtype="int64", missing="error")
)
template["/entry/events/event_time_offset"].copy_from(
    events["t"].convert_units("s", "us", dtype="float64")
)
template["/entry/events/event_time_offset"].attrs["units"] = "us"
template["/entry/duration"].set_from_result(events, "effective_duration")
template["/entry/duration"].attrs["units"] = "s"

template.validate().raise_if_errors()
print(template.summary())
report = mx.write_nexus(
    template, "converted.nxs", source="simulation/mccode.h5", entry_number=2,
    write_chunk_size=50_000, overwrite=False,
)
print(report.sources[events].n_selected)
destination = report.destinations["/entry/events/event_id"]
print(destination.shape, destination.dtype, destination.n_written)
for diagnostic in report.diagnostics:
    print(diagnostic.severity, diagnostic.path, diagnostic.message)
```

Apply these bindings to existing destinations in an original template. Reusing the
**same `EventSource` object** shares one execution, aligned rows, and statistics.
Separate objects execute independently even with equal settings; each write starts afresh.

Sampling reads hidden weights, preserves duplicates, and projects unit weights.
There is no time/pixel sorting; `ordered=True` means input order. Without `.sample(...)`,
rows stream unchanged. `filter_zeros=True` uses the source's `weight_key` even if not
output. `component_name=None` selects components with pixel IDs in reader order.

`id`/`t` alone do **not** establish complete `NXevent_data` pulse semantics: supply pulse
origins, event indexes, time references, and other metadata. Effective duration is selected
count / total input count rate, not inferred for arbitrary weights. Empty/no-positive samples
have no duration: use `set_from_result(events, callback)` with an explicit fallback or the
binding fails. Metadata validation cannot predict that case.

## Inspect and Clean

```python
reference = mx.create_template("experimental-run.nxs")
for retained in reference.retained_datasets(min_bytes=100 * 1024**2):
    print(retained.path, retained.aliases, retained.logical_bytes, retained.storage_bytes)
# Only if this group exists and you have decided its contents are obsolete:
del reference["/entry/old_events"]
print(reference.validate(retained_warning_bytes=100 * 1024**2).warnings)
print(reference.summary(limit=20))
```

Metadata-only audits count hard-linked objects once and show origin, shape, dtype, aliases,
and full/partial retention. Unknown sizes remain included at any `min_bytes`. Allocated
storage is not output size; partial updates report original byte counts. Audits reflect
edits; removing one alias may leave others. Empty templates have no records. Nothing auto-deletes.

Retention warnings are advisory (default 100 MiB/dataset); `retained_warning_bytes=None`
disables that threshold, not unknown-size reporting. First-release warnings are **diagnostics**
in `validate()`, `summary()`, and `report.diagnostics`, not Python `warnings`. Errors block writing.

`summary()` labels shared sources and shows ordered standard transforms, units, bindings,
and retention with bounded output. Custom callbacks remain opaque: names and `description`
labels are not verified mathematics. Neither method runs callbacks/sampling; their `source=`
does not inspect schemas (writer preflight does). Unknown sizes/statistics stay unknown.

## Source Access and Copies

`SourceContext(path, entry_number=2)` owns a full file path (default entry 1); use a
context manager. The writer closes owned inputs on success/failure.
`Read("simulation", filename="mccode.h5", entry_number=2)` takes a directory instead.
An open `Read`/`SourceContext` is borrowed with its selected entry: keep it open and close
it yourself. Conflicting entry overrides/closed handles fail. Static writes can omit `source`.

`node.set_from(callback, description="...")` passes a `SourceContext` for small calculations.
`ctx.raw` is file-wide; `ctx.mcstas`, `ctx.entry_name`, and `ctx.entry.raw` use the selected entry.
`ctx.read("/entry1/title")` preserves raw dtype; `ctx.entry.read_text("title")` decodes
entry-relative UTF-8 without parsing numbers. `selection=()` reads all; integers remove
axes, slices/tuples select axes. `iter_dataset(...)` batches the first retained axis, without tiling.

For an arbitrary root-absolute source dataset, assume `/entry` already exists:

```python
template.add_dataset("/entry/copied_values").copy_from(
    "/entry1/data/values", read_chunk_size=100_000, write_chunk_size=50_000,
    chunks=(4096,),  # Physical HDF5 chunks; this example assumes a 1D source.
)
```

Root-absolute paths ignore entry selection; the dataset must permit the chunk shape.
Ordinary/retained copies use bounded batches with multidimensional tiling, scalar/empty
support, not full materialization. Each ordinary-copy tile is limited by the read/write
batch size in elements; event batches count rows. Neither is a byte budget. Physical HDF5
`chunks`, compression, and `maxshape` are distinct. Whole bindings can replace shape/dtype;
`set_value(..., selection=...)` updates selections. Event-source bindings require whole
datasets, not selected projections. Later literal/callback/result selection patches follow
their recorded order. Source attributes do not replace destination attributes.

## Transforms and Memory

`EventColumn` chains `.scale(factor, dtype=...)`, `.offset(value, dtype=...)`,
`.map_values(mapping, dtype=..., missing="error")` (or `"keep"`), and
`.convert_units(from_unit, to_unit, dtype=...)`. Integer output rejects fractional values
and overflow; missing IDs are not rounded/dropped. Explicit time units: `s`, `ms`, `us`,
`ns`, `ps`; length: `m`, `cm`, `mm`, `um`, `nm`, `angstrom` (alias `A`). Unsupported pairs fail:
no unit inference, algebra, or new dependency. Conversions do not set attributes;
setting a `units` attribute does not convert values.

Single-column `.map(callback, dtype="float64", description="User time correction")` works.
Callbacks receive a column array, preserve row count/order, and must be deterministic and
batch-boundary independent. General multivariable custom chunk transforms are **not** in this delivery.

The reader loads each row block's **full table width** before projection. Sampling retains
**O(n_samples * variables)** values including hidden weights, plus copies/sort temporaries.
Output transforms are batched, but smaller writes do not bound sampler memory. Mapping
tables consume memory too. Replay sampling is **not implemented**. Seeds reproduce the
sampler only for identical inputs, settings, component order, filtering, and fixed read partitioning.
Read-batch changes may change samples/stats; write-batch changes do not change selection/alignment.

## Links and Publication

Use `template.move(old_path, new_path, repair_links=True)` for relocation and
`template.repair_links(old_path, new_path)` for incoming paths. Repairs cover resolvable
in-file soft links and absolute NeXus `target` metadata. Hard aliases preserve identity through
replacements; all aliases change. Delete/create links explicitly; conflicting alias edits fail.

Retained object/region references, virtual datasets (VDS), external raw storage,
null dataspaces, and non-text variable-length data are blocked: remove/fully replace
datasets and explicitly remove/replace reference attributes. Ordinary source copies also
reject indirect VDS/external raw storage and unsupported dtypes. Top-level HDF5 array
datatypes are unsupported; array-valued fields inside fixed compound dtypes are distinct.
Relative external filenames require an explicit absolute relink: after `import h5py`, delete and use
`template.create_link(path, h5py.ExternalLink("/absolute/dependency.nxs", "/target"))`.
External targets are not inspected; output depends on them. Cyclic groups and
ambiguous group-alias repairs are blocked. No arbitrary attribute/file rewriting.

```text
write_nexus(template, output, *, source=None, entry_number=None,
            read_chunk_size=100000, write_chunk_size=100000, overwrite=False,
            validators=(), progress=None)
```

Validators receive the open, completed output `h5py.File` before publication; raise on failure.
`progress(update)` gets `.phase`, `.path`, `.rows`; updates may be coarse, not percentages.
`report.sources[events]` holds `n_written`, `n_selected`, `total_weight`, `total_weight_variance`,
`effective_duration`, and `effective_duration_variance`. `report.destinations[path]` has
`.shape`, `.dtype`, `.n_written`. Result callbacks receive source stats, not an open output.

Writing rejects input/output aliases, requires `overwrite=True` for existing output, and
publishes atomically from a same-directory temporary file after result bindings/validators.
Failure cleans up, preserving existing output. Checks prevent overflow/truncation; compatible
inherited attributes/storage settings remain, with explicit correction or reported adaptations.
Alias protection includes resolved source datasets and retained external-link dependencies.
If temporary-file cleanup fails after publication, the completed write returns a cleanup
diagnostic rather than reporting a failed write; the diagnostic identifies the leftover path.
