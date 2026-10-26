# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 McCode-dev contributors (https://github.com/mccode-dev)
"""Template instructions are inspectable without reading or writing payloads."""

from dataclasses import dataclass

import h5py
import numpy as np
import pytest

from mcstastox.NexusTemplate import (
    Binding,
    Diagnostic,
    NexusTemplate,
    NodePlan,
    ValidationReport,
    create_empty_template,
    create_template,
)


@pytest.fixture
def base_file(tmp_path):
    path = tmp_path / "base.nxs"
    with h5py.File(path, "w") as file:
        file.attrs["creator"] = "facility"
        entry = file.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"
        data = entry.create_dataset("data", data=np.arange(12).reshape(4, 3))
        data.attrs["units"] = "counts"
        data.attrs["obsolete"] = 1
        entry.create_dataset("title", data=np.bytes_("old"))
        entry.create_dataset("fixed", data=np.array([b"abc"], dtype="S3"))
        entry.create_dataset("text", data=["unknown length"], dtype=h5py.string_dtype())
        other = file.create_group("other")
        other["alias"] = data
        entry["soft"] = h5py.SoftLink("data")
    return path


@pytest.fixture
def no_payload_reads(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("template inspection must not read dataset payloads")

    monkeypatch.setattr(h5py.Dataset, "__getitem__", forbidden)
    monkeypatch.setattr(h5py.Dataset, "__array__", forbidden)
    monkeypatch.setattr(h5py.Dataset, "read_direct", forbidden)


@pytest.fixture(params=[3, None])
def utf8_template(request, tmp_path, no_payload_reads):
    path = tmp_path / "utf8-template.nxs"
    with h5py.File(path, "w") as file:
        file.create_dataset(
            "text", shape=(1,), dtype=h5py.string_dtype("utf-8", request.param)
        )
    return create_template(path)


@dataclass(eq=False)
class SourceSpec:
    description: object = "Detector, weighted sample, requested=100, seed=42"
    sampling: object = "metadata only"

    def iter_chunks(self, *args):
        pytest.fail("template inspection must not execute providers")


@dataclass(eq=False)
class ColumnSpec:
    source: SourceSpec
    description: str


def test_constructors_and_repr_are_lazy(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("constructors and repr must not open or create files")

    monkeypatch.setattr(h5py, "File", forbidden)
    backed = create_template(tmp_path / "not-created.nxs")
    empty = create_empty_template()
    assert isinstance(backed, NexusTemplate)
    assert isinstance(empty, NexusTemplate)
    assert backed.base_path == tmp_path / "not-created.nxs"
    assert empty.base_path is None
    assert "not-created.nxs" in repr(backed)
    assert "base_path=None" in repr(empty)
    assert len(repr(backed)) < 300
    assert backed._objects == {}
    assert empty.retained_datasets() == []
    assert not empty.validate().diagnostics
    assert set(empty._snapshot()) == {"/"}
    assert "base=empty" in empty.summary()


def test_empty_layout_and_intentionally_empty_datasets(no_payload_reads):
    template = create_empty_template()
    entry = template.add_group("/entry")
    entry.attrs["NX_class"] = "NXentry"
    entry.add_dataset("title", value="Simulated events")
    events = entry.add_group("events")
    events.attrs["NX_class"] = "NXevent_data"
    dataset = events.add_dataset("id", shape=(0,), dtype="int64", maxshape=(None,))
    assert dataset.describe("Aligned detector IDs") is dataset
    assert "Aligned detector IDs" in template.summary()
    snapshot = template._snapshot()
    assert snapshot["/"].kind == "group"
    assert snapshot["/entry/events/id"].creation == {
        "shape": (0,),
        "dtype": np.dtype("int64"),
        "maxshape": (None,),
    }
    assert snapshot["/entry/events/id"].bindings == []
    assert snapshot["/entry/title"].bindings[0].value == "Simulated events"
    assert snapshot["/entry"].attributes["NX_class"].kind == "literal"
    assert template.retained_datasets() == []
    template.validate().raise_if_errors()


def test_targeted_lookup_does_not_materialize_unrelated_tree(
    base_file, no_payload_reads
):
    template = create_template(base_file)
    title = template["/entry/title"]
    assert repr(title) == "NodeProxy('/entry/title')"
    assert title.kind == "dataset"
    loaded = {node.base_path for node in template._objects.values()}
    assert loaded == {"/", "/entry", "/entry/title"}


def test_snapshot_contract_and_copied_original_attributes(base_file, no_payload_reads):
    template = create_template(base_file)
    snapshot = template._snapshot()
    assert all(isinstance(plan, NodePlan) for plan in snapshot.values())
    assert all(path == plan.path for path, plan in snapshot.items())
    assert snapshot["/"].attributes["creator"].value == "facility"
    data, alias = snapshot["/entry/data"], snapshot["/other/alias"]
    with h5py.File(base_file, "r") as file:
        assert data.object_key == h5py.h5o.get_info(file["entry/data"].id).addr
    assert data.object_key == alias.object_key
    assert data.creation is alias.creation
    assert data.attributes is alias.attributes
    assert data.bindings is alias.bindings
    assert data.base_path == "/entry/data"
    assert data.creation["shape"] == (4, 3)
    assert snapshot["/entry/soft"].kind == "soft"
    assert snapshot["/entry/soft"].link.path == "data"
    template.validate().raise_if_errors()


def test_bindings_are_explicit_fluent_and_selection_updates_are_ordered(
    base_file, no_payload_reads
):
    template = create_template(base_file)
    target = template["/entry/data"]

    def callback(source):
        pytest.fail("do not invoke the source callback")

    assert target.set_from(callback, description="Small calculated array") is target
    assert target.set_value(3, selection=(0, slice(None))) is target
    assert (
        target.copy_from(
            "/entry2/data",
            selection=(1, slice(None)),
            read_chunk_size=7,
            write_chunk_size=3,
            columns=("id", "t"),
            dtype="float64",
        )
        is target
    )
    source = SourceSpec()
    assert target.set_from_result(source, "n_selected", selection=(2, 0)) is target
    bindings = template._snapshot()[target.path].bindings
    assert [b.kind for b in bindings] == ["callback", "literal", "copy", "result"]
    assert bindings[0].value is callback
    assert bindings[0].selection is None
    assert bindings[0].options == {"description": "Small calculated array"}
    assert bindings[1].selection == (0, slice(None))
    assert bindings[2].value == "/entry2/data"
    assert bindings[2].options == {
        "read_chunk_size": 7,
        "write_chunk_size": 3,
        "columns": ("id", "t"),
    }
    assert bindings[3].value == (source, "n_selected")
    assert template._snapshot()[target.path].creation["dtype"] == np.dtype("float64")
    assert target.set_value(np.ones((2, 3))) is target
    assert len(template._snapshot()[target.path].bindings) == 1
    template.validate().raise_if_errors()


def test_attributes_are_literal_unless_explicitly_bound(base_file, no_payload_reads):
    template = create_template(base_file)
    target = template["/entry/data"]
    attrs = target.attrs
    assert "units" in attrs
    assert "missing" not in attrs
    attrs["units"] = "us"
    del attrs["obsolete"]

    def callback(source):
        pytest.fail("do not invoke attribute callbacks")

    source = SourceSpec()
    attr = attrs["calculation"]
    assert attr.set_from(callback, description="Chosen units") is attr
    assert (
        attrs["duration"].set_from_result(source, "effective_duration").name
        == "duration"
    )
    attrs["custom"].set_from_result(source, callback, description="Custom statistic")
    attrs["literal_callable"] = callback
    attributes = template._snapshot()[target.path].attributes
    assert attributes["units"].value == "us"
    assert "obsolete" not in attributes
    assert attributes["calculation"].kind == "callback"
    assert attributes["duration"].value == (source, "effective_duration")
    assert attributes["custom"].value[1] is callback
    assert attributes["literal_callable"].kind == "literal"
    assert attrs["units"].binding is attributes["units"]
    assert set(attrs) == set(attributes)
    assert len(attrs) == len(attributes)
    assert any(
        "literal attribute dtype" in d.message for d in template.validate().errors
    )
    del attrs["literal_callable"]
    template.validate().raise_if_errors()


@pytest.mark.parametrize("labels", [("alpha", "caf\u00e9"), ()])
def test_retained_object_text_attribute_arrays(
    tmp_path, no_payload_reads, labels, monkeypatch
):
    path = tmp_path / "text-attributes.nxs"
    with h5py.File(path, "w") as file:
        file.attrs["labels"] = np.array(labels, dtype=h5py.string_dtype())
        file.create_dataset("unread", shape=(4,), dtype="int64")
    original_getitem = h5py.AttributeManager.__getitem__

    def untagged_getitem(attrs, name):
        value = original_getitem(attrs, name)
        if isinstance(value, np.ndarray) and value.dtype.kind == "O":
            untagged = np.empty(value.shape, dtype=object)
            untagged[...] = value
            return untagged
        return value

    monkeypatch.setattr(h5py.AttributeManager, "__getitem__", untagged_getitem)
    template = create_template(path)
    binding = template._snapshot()["/"].attributes["labels"]
    assert binding.kind == "literal"
    assert binding.value.dtype.kind == "O"
    assert h5py.check_string_dtype(binding.value.dtype).encoding == "utf-8"
    np.testing.assert_array_equal(binding.value, np.array(labels, dtype=object))
    template.validate().raise_if_errors()
    assert "@labels" in template.summary()


@pytest.mark.parametrize(
    "labels",
    [
        np.array(["alpha", "caf\u00e9"], dtype=object),
        np.array([b"alpha", b"caf\xc3\xa9"], dtype=object),
        np.array(
            [["alpha", b"caf\xc3\xa9"], [np.str_("caf\u00e9"), np.bytes_(b"alpha")]],
            dtype=object,
        ),
        np.array("caf\u00e9", dtype=object),
        np.array([], dtype=h5py.string_dtype()),
        np.array([["alpha"], ["caf\u00e9"]], dtype=h5py.string_dtype()),
    ],
)
def test_literal_object_text_arrays_preserve_values_and_infer_utf8(
    labels, no_payload_reads
):
    template = create_empty_template()
    original_dtype = labels.dtype
    template.attrs["labels"] = labels
    template.add_dataset("/labels", value=labels).set_value(labels)
    snapshot = template._snapshot()
    assert snapshot["/"].attributes["labels"].value is labels
    assert snapshot["/labels"].bindings[0].value is labels
    assert snapshot["/labels"].creation["shape"] == labels.shape
    string = h5py.check_string_dtype(snapshot["/labels"].creation["dtype"])
    assert string is not None
    assert string.encoding == "utf-8"
    template.validate().raise_if_errors()
    assert labels.dtype is original_dtype
    assert "@labels" in template.summary()


@pytest.mark.parametrize(
    "labels",
    [
        np.array([1, 2], dtype=object),
        np.array(["alpha", 1], dtype=object),
        np.array([object()], dtype=object),
        np.array([None], dtype=object),
        np.array([], dtype=object),
        np.array([h5py.Reference()], dtype=object),
        np.array([h5py.RegionReference()], dtype=object),
        np.array(["alpha", h5py.Reference()], dtype=object),
        np.array([], dtype=h5py.ref_dtype),
        np.array(["alpha"], dtype=h5py.ref_dtype),
        np.array([1, 2], dtype=h5py.string_dtype()),
        np.array([object()], dtype=h5py.string_dtype()),
        np.array(["alpha"], dtype=h5py.vlen_dtype(np.dtype("int32"))),
    ],
)
def test_object_arrays_are_not_implicitly_coerced_to_text(labels, no_payload_reads):
    template = create_empty_template()
    template.attrs["labels"] = labels
    template.add_dataset("/labels", value=labels)
    report = template.validate()
    assert any(d.path == "/@labels" for d in report.errors)
    assert any(d.path == "/labels" for d in report.errors)


def test_explicit_creation_keys_remain_available_to_writer(base_file, no_payload_reads):
    template = create_template(base_file)
    inherited = template._snapshot()["/entry/data"]
    assert template._objects[inherited.object_key].explicit_creation == set()
    template["/entry/data"].copy_from("/source/data", dtype="float32", chunks=(2, 3))
    template.add_dataset("/inferred", value=[1, 2])
    template.add_dataset("/explicit", shape=(0,), dtype="int64")
    snapshot = template._snapshot()
    assert template._objects[snapshot["/entry/data"].object_key].explicit_creation == {
        "dtype",
        "chunks",
    }
    assert template._objects[snapshot["/other/alias"].object_key].explicit_creation == {
        "dtype",
        "chunks",
    }
    assert (
        template._objects[snapshot["/inferred"].object_key].explicit_creation == set()
    )
    assert template._objects[snapshot["/explicit"].object_key].explicit_creation == {
        "shape",
        "dtype",
    }


def test_additions_deletions_parent_conflicts_and_original_file_unchanged(
    base_file, no_payload_reads
):
    original = base_file.read_bytes()
    template = create_template(base_file)
    del template["/entry/fixed"]
    template.add_group("/entry/analysis").attrs["NX_class"] = "NXcollection"
    template.add_dataset("/entry/analysis/value", value=42.0, dtype="float32")
    with pytest.raises(ValueError, match="already exists"):
        template.add_dataset("/entry/analysis/value", value=1)
    with pytest.raises(ValueError, match="already exists"):
        template.add_group("/entry")
    with pytest.raises(KeyError):
        template.add_group("/missing/child")
    with pytest.raises(TypeError, match="not a group"):
        template.add_group("/entry/title/child")
    with pytest.raises(TypeError, match="not a dataset"):
        template["/entry"].set_value(1)
    with pytest.raises(ValueError, match="root"):
        del template["/"]
    with pytest.raises(KeyError):
        del template["/missing"]
    assert "/entry/fixed" not in template
    assert "/entry/analysis/value" in template
    assert "/entry/title/child" not in template
    assert "Removed: /entry/fixed" in template.summary()
    template.validate().raise_if_errors()
    assert base_file.read_bytes() == original


def test_retained_sizes_aliases_and_unknown_threshold(base_file, no_payload_reads):
    template = create_template(base_file)
    records = {r.path: r for r in template.retained_datasets()}
    assert len(records) == 4
    data = records["/entry/data"]
    assert data.logical_bytes == 12 * np.dtype("int64").itemsize
    assert data.storage_bytes == data.logical_bytes
    assert data.aliases == ("/other/alias",)
    assert not data.partial
    assert data.base_file == base_file
    assert data.origin_file == base_file
    assert data.base_path == data.origin_path == "/entry/data"
    assert data.shape == (4, 3)
    assert data.dtype == np.dtype("int64")
    unknown = template.retained_datasets(min_bytes=10**12)
    assert len(unknown) == 1
    assert unknown[0].path == "/entry/text"
    assert unknown[0].logical_bytes is None
    assert "variable-length" in unknown[0].unknown_reason
    assert "unknown size: /entry/text" in template.summary()


def test_retention_reflects_cleanup_moves_partial_updates_and_replacements(
    base_file, no_payload_reads
):
    template = create_template(base_file)
    template["/entry/data"].set_value(99, selection=(0, 0))
    records = template.retained_datasets()
    assert records[0].partial
    assert records[0].logical_bytes == 96
    del template["/entry/data"]
    data = next(r for r in template.retained_datasets() if r.path == "/other/alias")
    assert data.aliases == ()
    assert data.partial
    template.move("/other/alias", "/other/moved", repair_links=True)
    data = next(r for r in template.retained_datasets() if r.path == "/other/moved")
    assert data.base_path == "/entry/data"
    template["/other/moved"].set_from(lambda source: pytest.fail("not executed"))
    del template["/entry/fixed"]
    del template["/entry/title"]
    assert [r.path for r in template.retained_datasets()] == ["/entry/text"]
    del template["/entry"]
    assert template.retained_datasets() == []


def test_full_replacement_preserves_hard_alias_identity(base_file, no_payload_reads):
    template = create_template(base_file)
    template["/other/alias"].set_value([1, 2], dtype="uint8")
    template["/entry/data"].attrs["units"] = "new units"
    snapshot = template._snapshot()
    first, alias = snapshot["/entry/data"], snapshot["/other/alias"]
    assert first.object_key == alias.object_key
    assert first.bindings is alias.bindings
    assert first.bindings[0].value == [1, 2]
    assert first.creation["shape"] == (2,)
    assert alias.attributes["units"].value == "new units"
    assert "/entry/data" not in [r.path for r in template.retained_datasets()]
    report = template.validate()
    assert not report.errors
    assert any("all hard-link aliases" in d.message for d in report.warnings)


def test_conflicting_alias_replacements_and_attributes_are_errors(
    base_file, no_payload_reads
):
    template = create_template(base_file)
    template["/entry/data"].set_value([1, 2])
    template["/other/alias"].set_value([3, 4])
    template["/entry/data"].attrs["units"] = "s"
    template["/other/alias"].attrs["units"] = "ms"
    report = template.validate()
    assert any(
        "Conflicting alias edits" in d.message and "binding" in d.message
        for d in report.errors
    )
    assert any("attribute:units" in d.message for d in report.errors)
    with pytest.raises(ValueError, match="Conflicting alias edits"):
        report.raise_if_errors()


def test_disjoint_alias_selection_updates_are_supported(base_file, no_payload_reads):
    template = create_template(base_file)
    template["/entry/data"].set_value(1, selection=(0, 0))
    template["/other/alias"].set_value(2, selection=(1, 0))
    assert not template.validate().errors
    template["/other/alias"].set_value(3, selection=(0, 0))
    assert any(
        "Conflicting alias edits" in d.message for d in template.validate().errors
    )


def test_retained_size_warnings_are_advisory_not_python_warnings(
    base_file, no_payload_reads
):
    template = create_template(base_file)
    report = template.validate(retained_warning_bytes=90)
    report.raise_if_errors()
    large = [d for d in report.warnings if "Large inherited" in d.message]
    assert len(large) == 1
    assert large[0].path == "/entry/data"
    assert not any("Large inherited" in d.message for d in template.validate().warnings)
    assert not any(
        "Large inherited" in d.message
        for d in template.validate(retained_warning_bytes=None).warnings
    )
    assert "Large inherited" not in template.summary()
    assert len(template.retained_datasets()) == 4


def test_sparse_compressed_storage_is_not_confused_with_logical_size(
    tmp_path, no_payload_reads
):
    path = tmp_path / "sparse.nxs"
    with h5py.File(path, "w") as file:
        file.create_dataset(
            "large", shape=(10**9,), dtype="int64", chunks=(1000,), compression="gzip"
        )
    template = create_template(path)
    record = template.retained_datasets()[0]
    assert record.logical_bytes == 8 * 10**9
    assert record.storage_bytes == 0
    assert any("Large inherited" in d.message for d in template.validate().warnings)
    del template["/large"]
    assert not template.validate().warnings
    assert template.retained_datasets() == []


def test_summary_is_opaque_bounded_and_tracks_shared_source_identity(no_payload_reads):
    template = create_empty_template()
    first, separate = SourceSpec(), SourceSpec()

    def correction(source):
        pytest.fail("summary/validation must not invoke named callbacks")

    template.add_dataset("/a", shape=(0,), dtype="int64").copy_from(
        ColumnSpec(first, "id -> value mapping (10000 entries, missing=error)")
    )
    template.add_dataset("/b", shape=(0,), dtype="float64").copy_from(
        ColumnSpec(first, "t -> convert s to us (multiply by 1000000)")
    )
    template.add_dataset("/duration", shape=(), dtype="float64").set_from_result(
        first, "effective_duration"
    )
    template.add_dataset("/other", shape=(0,), dtype="int64").copy_from(
        ColumnSpec(separate, "id")
    )
    template.add_dataset("/named", shape=(), dtype="int64").set_from(correction)
    template.add_dataset("/anonymous", shape=(), dtype="int64").set_from(
        lambda source: 2
    )
    template.add_dataset("/labelled", shape=(), dtype="int64").set_from(
        correction, description="User temperature correction"
    )
    text = template.summary(source=object())
    assert "sample1:" in text
    assert "sample2:" in text
    assert text.count("sample1 result -> effective_duration") == 1
    assert text.count("sample1 ->") == 2
    assert "value mapping (10000 entries" in text
    assert "convert s to us" in text
    assert "custom callback correction" in text
    assert "custom callback" in text
    assert "<lambda>" not in text
    assert "user description: User temperature correction" in text
    template.validate(source=object()).raise_if_errors()
    for index in range(100):
        template.add_dataset(f"/extra{index}", value=index)
    bounded = template.summary(limit=2)
    assert len(bounded) < 2500
    assert "more nodes" in bounded
    assert "/extra99:" not in bounded


def test_summary_does_not_render_an_entire_provider_mapping(no_payload_reads):
    class BoundedMapping(dict):
        def __repr__(self):
            pytest.fail("must not stringify the full provider mapping")

    template = create_empty_template()
    provider = SourceSpec(
        description=BoundedMapping((f"key{x}", x) for x in range(1000))
    )
    template.add_dataset("/value", shape=(0,), dtype="int64").copy_from(provider)
    text = template.summary(limit=2)
    assert "1000 entries" in text
    assert "key999" not in text
    assert len(text) < 2000


def test_real_provider_descriptions_show_transforms_and_sampling_without_execution(
    no_payload_reads, monkeypatch
):
    from mcstastox.NexusSource import EventColumn, EventSource, event_source
    from mcstastox.Sampling import SamplingSettings

    def forbidden(*args, **kwargs):
        pytest.fail("provider execution is not part of template inspection")

    monkeypatch.setattr(EventSource, "iter_chunks", forbidden)
    monkeypatch.setattr(EventSource, "schema", forbidden)
    monkeypatch.setattr(EventColumn, "apply", forbidden)
    events = event_source(component_name="Detector", variables=("id", "t")).sample(
        SamplingSettings(n_samples=100, seed=42)
    )
    template = create_empty_template()
    template.add_dataset("/ids", shape=(0,), dtype="int64").copy_from(
        events["id"].map_values({0: 100, 1: 101}, dtype="int64")
    )
    time = template.add_dataset("/time", shape=(0,), dtype="float64").copy_from(
        events["t"].convert_units("s", "us").offset(1)
    )
    time.attrs["units"] = "s"
    template.add_dataset("/duration", shape=(), dtype="float64").set_from_result(
        events, "effective_duration"
    )
    text = template.summary()
    assert "sample1: weighted sample requested=100, seed=42" in text
    assert "Detector" in text
    assert "sample1 -> id -> value mapping (2 entries, missing=error)" in text
    assert "sample1 -> t -> convert s to us" in text
    assert text.index("convert s to us") < text.index("offset by 1")
    assert "sample1 result -> effective_duration" in text
    report = template.validate(source=object())
    assert not report.errors
    assert any("conflict with destination units" in d.message for d in report.warnings)


def test_moving_an_edited_object_does_not_create_false_alias_conflicts(
    no_payload_reads,
):
    template = create_empty_template()
    template.add_dataset("/old", value=1).attrs["units"] = "s"
    template["/old"].set_value(2)
    template.move("/old", "/new", repair_links=True)
    template["/new"].set_value(3)
    template["/new"].attrs["units"] = "ms"
    template.validate().raise_if_errors()


def test_result_fields_are_names_not_expressions(no_payload_reads):
    template = create_empty_template()
    provider = SourceSpec()
    target = template.add_dataset("/duration", shape=(), dtype="float64")
    target.set_from_result(provider, "effective_duration * 2")
    assert any("Unknown result field" in d.message for d in template.validate().errors)
    target.set_from_result(provider, lambda result: 2, description="Double duration")
    template.validate().raise_if_errors()
    assert "user description: Double duration" in template.summary()


@pytest.mark.parametrize(
    ("selection", "value"),
    [
        ((1, slice(None)), [1, 2, 3]),
        ((slice(0, 2), slice(0, 2)), 1),
        ((Ellipsis, 0), [1, 2, 3, 4]),
        ((-1, -1), 1),
        ((slice(1, 1), slice(None)), np.empty((0, 3), dtype="int64")),
    ],
)
def test_basic_selection_shapes(base_file, no_payload_reads, selection, value):
    template = create_template(base_file)
    template["/entry/data"].set_value(value, selection=selection)
    template.validate().raise_if_errors()


@pytest.mark.parametrize(
    ("selection", "value", "message"),
    [
        ((4, 0), 1, "out of bounds"),
        ((0, 0, 0), 1, "too many indices"),
        ((slice(None, None, -1), 0), 1, "positive"),
        (([0, 1], 0), 1, "only integers"),
        ((Ellipsis, Ellipsis), 1, "more than one ellipsis"),
        ((0, slice(None)), [1, 2], "broadcast"),
    ],
)
def test_invalid_selections_are_diagnostics(
    base_file, no_payload_reads, selection, value, message
):
    template = create_template(base_file)
    template["/entry/data"].set_value(value, selection=selection)
    assert any(message in d.message for d in template.validate().errors)


@pytest.mark.parametrize(
    ("kind", "creation"),
    [
        ("callback", {}),
        ("callback", {"shape": (), "dtype": "int64"}),
        ("copy", {"shape": (0,), "dtype": "int64"}),
        ("provider", {"shape": (0, 2), "dtype": "int64"}),
        ("result", {}),
        ("result", {"shape": (0,), "dtype": "int64"}),
    ],
)
def test_dynamic_whole_binding_defers_selected_shape_checks(
    kind, creation, no_payload_reads
):
    def forbidden(*args):
        pytest.fail("template validation must not execute deferred bindings")

    template = create_empty_template()
    target = template.add_dataset("/dynamic", **creation)
    if kind == "callback":
        target.set_from(forbidden)
    elif kind == "copy":
        target.copy_from("/entry1/data")
    elif kind == "provider":
        target.copy_from(SourceSpec())
    else:
        target.set_from_result(SourceSpec(), forbidden)
    target.set_value(99, selection=0)
    target.set_value([7, 8, 9], selection=(1, slice(None)))
    target.set_value(3, selection=(Ellipsis, -1))
    plan = template._snapshot()["/dynamic"]
    assert plan.creation.get("shape") == creation.get("shape")
    assert len(plan.bindings) == 4
    assert plan.bindings[1].selection == 0
    template.validate(source=object()).raise_if_errors()
    assert "error /dynamic" not in template.summary(source=object())


@pytest.mark.parametrize(
    ("selection", "message"),
    [
        ([0, 1], "only integers"),
        ((Ellipsis, Ellipsis), "more than one ellipsis"),
        (slice(None, None, -1), "positive"),
        (slice(None, None, 0), "positive"),
        (slice(0, 2.5), "slice indices"),
        ((None,), "only integers"),
        (True, "only integers"),
    ],
)
def test_dynamic_whole_binding_still_validates_selection_syntax(
    selection, message, no_payload_reads
):
    def forbidden(source):
        pytest.fail("template validation must not execute callbacks")

    template = create_empty_template()
    target = template.add_dataset("/dynamic").set_from(forbidden)
    target.set_value(1, selection=selection)
    assert any(message in d.message for d in template.validate().errors)


def test_dynamic_whole_binding_still_validates_literal_dtype(no_payload_reads):
    template = create_empty_template()
    target = template.add_dataset("/dynamic", shape=(0,), dtype="uint8")
    target.copy_from("/entry1/data").set_value(256, selection=0)
    assert any("overflows" in d.message for d in template.validate().errors)


@pytest.mark.parametrize(
    ("selection", "value", "message"),
    [
        ((2, 0), 1, "out of bounds"),
        ((0, slice(None)), [1, 2], "broadcast"),
    ],
)
def test_literal_whole_replacement_restores_strict_selection_validation(
    selection, value, message, no_payload_reads
):
    def forbidden(source):
        pytest.fail("superseded callbacks must not be executed")

    template = create_empty_template()
    target = template.add_dataset("/known").set_from(forbidden)
    target.set_value(np.zeros((2, 3), dtype="int64"))
    target.set_value(value, selection=selection)
    assert any(message in d.message for d in template.validate().errors)


@pytest.mark.parametrize("shape", [None, (0,)])
def test_selection_without_dynamic_whole_still_requires_valid_shape(
    shape, no_payload_reads
):
    template = create_empty_template()
    target = template.add_dataset("/known", shape=shape, dtype="int64")
    target.set_value(1, selection=0)
    message = "known destination shape" if shape is None else "out of bounds"
    assert any(message in d.message for d in template.validate().errors)


def test_scalar_and_length_one_strings_stay_distinct(base_file, no_payload_reads):
    template = create_template(base_file)
    template["/entry/title"].set_value("a much longer title")
    template["/entry/fixed"].set_value("new", selection=0)
    snapshot = template._snapshot()
    assert snapshot["/entry/title"].creation["shape"] == ()
    assert (
        h5py.check_string_dtype(snapshot["/entry/title"].creation["dtype"]).encoding
        == "utf-8"
    )
    assert snapshot["/entry/fixed"].creation["shape"] == (1,)
    template.validate().raise_if_errors()
    template["/entry/title"].set_value("new", selection=0)
    assert any("too many indices" in d.message for d in template.validate().errors)


def test_fixed_width_string_truncation_is_rejected(base_file, no_payload_reads):
    template = create_template(base_file)
    template["/entry/fixed"].set_value("too long", selection=0)
    assert any("truncate" in d.message for d in template.validate().errors)
    template["/entry/fixed"].set_value(["long replacement"])
    template.validate().raise_if_errors()
    assert "/entry/fixed" not in [r.path for r in template.retained_datasets()]


def test_whole_literal_text_replacement_retains_utf8(utf8_template):
    value = np.array([b"caf\xc3\xa9"], dtype="S5")
    utf8_template["/text"].set_value(value)
    plan = utf8_template._snapshot()["/text"]
    string = h5py.check_string_dtype(plan.creation["dtype"])
    assert string.encoding == "utf-8"
    assert string.length == 5
    assert plan.bindings[0].value is value
    assert h5py.check_string_dtype(value.dtype).encoding == "ascii"
    utf8_template.validate().raise_if_errors()
    assert utf8_template.retained_datasets() == []


@pytest.mark.parametrize(
    ("dtype", "encoding", "length"),
    [
        ("S5", "ascii", 5),
        (h5py.string_dtype("ascii"), "ascii", None),
        (h5py.string_dtype("utf-8", 8), "utf-8", 8),
    ],
)
def test_whole_literal_text_replacement_honors_explicit_dtype(
    utf8_template, dtype, encoding, length
):
    value = np.array([b"alpha"], dtype="S5")
    utf8_template["/text"].set_value(value, dtype=dtype)
    plan = utf8_template._snapshot()["/text"]
    string = h5py.check_string_dtype(plan.creation["dtype"])
    assert string.encoding == encoding
    assert string.length == length
    assert "dtype" in utf8_template._objects[plan.object_key].explicit_creation
    utf8_template.validate().raise_if_errors()


def test_whole_literal_text_replacement_rejects_explicit_utf8_truncation(utf8_template):
    utf8_template["/text"].set_value(
        np.array([b"caf\xc3\xa9"], dtype="S5"), dtype=h5py.string_dtype("utf-8", 3)
    )
    string = h5py.check_string_dtype(
        utf8_template._snapshot()["/text"].creation["dtype"]
    )
    assert string.encoding == "utf-8"
    assert string.length == 3
    assert any("truncate" in d.message for d in utf8_template.validate().errors)


def test_literal_selection_does_not_reinfer_utf8_dtype(utf8_template):
    dtype = utf8_template._snapshot()["/text"].creation["dtype"]
    utf8_template["/text"].set_value(
        np.array([b"caf\xc3\xa9"], dtype="S5"), selection=slice(None)
    )
    string = h5py.check_string_dtype(
        utf8_template._snapshot()["/text"].creation["dtype"]
    )
    original = h5py.check_string_dtype(dtype)
    assert string == original
    if original.length is None:
        utf8_template.validate().raise_if_errors()
    else:
        assert any("truncate" in d.message for d in utf8_template.validate().errors)


def test_whole_literal_replacement_preserves_explicitly_created_utf8(no_payload_reads):
    template = create_empty_template()
    target = template.add_dataset(
        "/text", shape=(1,), dtype=h5py.string_dtype("utf-8", 3)
    )
    target.set_value(np.array([b"caf\xc3\xa9"], dtype="S5"))
    string = h5py.check_string_dtype(template._snapshot()["/text"].creation["dtype"])
    assert string.encoding == "utf-8"
    assert string.length == 5
    template.validate().raise_if_errors()


def test_whole_literal_numeric_replacement_does_not_keep_text_dtype(utf8_template):
    utf8_template["/text"].set_value([42])
    dtype = utf8_template._snapshot()["/text"].creation["dtype"]
    assert dtype == np.dtype("int64")
    assert h5py.check_string_dtype(dtype) is None
    utf8_template.validate().raise_if_errors()


def test_utf8_literals_and_embedded_nul_validation(no_payload_reads):
    template = create_empty_template()
    template.add_dataset("/text", value="\u03bcs")
    template.validate().raise_if_errors()
    template["/text"].set_value("bad\0text")
    assert any("embedded NUL" in d.message for d in template.validate().errors)


@pytest.mark.parametrize(
    ("value", "dtype", "message"),
    [
        (256, "uint8", "overflows"),
        (-1, "uint8", "overflows"),
        (1.5, "int64", "non-integral"),
        (float("nan"), "int64", "non-integral"),
        ("not numeric", "int64", "numeric"),
        ("long", "S2", "truncate"),
        (np.uint64(2**63), "int64", "overflows"),
        (float(2**63), "int64", "overflows"),
        (1 + 2j, "float64", "complex literals"),
    ],
)
def test_unsafe_literal_casts_are_rejected(value, dtype, message, no_payload_reads):
    template = create_empty_template()
    template.add_dataset("/value", value=value, dtype=dtype)
    assert any(message in d.message for d in template.validate().errors)


def test_selection_float_precision_requires_explicit_dtype(tmp_path, no_payload_reads):
    path = tmp_path / "float.nxs"
    with h5py.File(path, "w") as file:
        file.create_dataset("value", shape=(1,), dtype="float32")
    template = create_template(path)
    template["/value"].set_value(1.25, selection=0)
    assert any("unsafe implicit cast" in d.message for d in template.validate().errors)
    template = create_template(path)
    template["/value"].set_value(1.25, selection=0, dtype="float32")
    template.validate().raise_if_errors()


def test_compatible_creation_properties_are_preserved_and_incompatible_ones_block(
    tmp_path, no_payload_reads
):
    path = tmp_path / "chunked.nxs"
    with h5py.File(path, "w") as file:
        file.create_dataset(
            "value",
            shape=(10, 2),
            dtype="float64",
            chunks=(5, 2),
            maxshape=(None, 2),
            compression="gzip",
            compression_opts=2,
            shuffle=True,
            fletcher32=True,
            fillvalue=-1,
        )
    template = create_template(path)
    spec = template._snapshot()["/value"].creation
    assert spec["chunks"] == (5, 2)
    assert spec["compression"] == "gzip"
    assert spec["compression_opts"] == 2
    assert spec["maxshape"] == (None, 2)
    assert spec["shuffle"] is True
    assert spec["fletcher32"] is True
    assert spec["fillvalue"] == -1
    template["/value"].set_value(1.0)
    assert any("Scalar datasets" in d.message for d in template.validate().errors)
    template["/value"].set_value(
        1.0,
        chunks=None,
        maxshape=None,
        compression=None,
        compression_opts=None,
        shuffle=False,
        fletcher32=False,
    )
    template.validate().raise_if_errors()


@pytest.mark.parametrize("replacement", ["delete", "literal", "callback", "copy"])
def test_retained_top_level_array_dtype_is_blocked_until_removed_or_replaced(
    tmp_path, no_payload_reads, replacement
):
    path = tmp_path / "array-dtype.nxs"
    dtype = np.dtype(("int16", (2,)))
    with h5py.File(path, "w") as file:
        dataset = file.create_dataset("array", shape=(3,), dtype=dtype)
        dataset.attrs["units"] = "counts"
        file["alias"] = dataset
    template = create_template(path)
    report = template.validate()
    assert any("top-level array" in d.message.lower() for d in report.errors)
    retained = template.retained_datasets()[0]
    assert retained.dtype == dtype
    assert retained.logical_bytes == 3 * dtype.itemsize
    assert len(retained.aliases) == 1
    template["/array"].set_value(1, selection=0)
    assert any(
        "top-level array" in d.message.lower() for d in template.validate().errors
    )

    def forbidden(source):
        pytest.fail("replacement callbacks must not execute during validation")

    if replacement == "delete":
        del template["/array"]
        del template["/alias"]
    elif replacement == "literal":
        template["/array"].set_value(np.array([1, 2, 3], dtype="int16"))
        assert template._snapshot()["/array"].creation["dtype"].subdtype is None
    elif replacement == "callback":
        template["/array"].set_from(forbidden)
        assert "dtype" not in template._snapshot()["/array"].creation
    else:
        template["/array"].copy_from("/entry1/data")
        assert "dtype" not in template._snapshot()["/array"].creation
    template.validate().raise_if_errors()
    assert template.retained_datasets() == []


@pytest.mark.parametrize("binding", ["empty", "literal", "callback"])
def test_explicit_top_level_array_output_dtype_is_unsupported(
    binding, no_payload_reads
):
    template = create_empty_template()
    dtype = np.dtype(("int16", (2,)))
    target = template.add_dataset("/array", shape=(3,), dtype=dtype)
    if binding == "literal":
        target.set_value([1, 2, 3], dtype=dtype)
    elif binding == "callback":
        target.set_from(lambda source: pytest.fail("callbacks must not execute"))
    assert any(
        d.path == "/array" and "top-level array" in d.message.lower()
        for d in template.validate().errors
    )


def test_compound_dtype_array_fields_are_not_top_level_array_types(
    tmp_path, no_payload_reads
):
    path = tmp_path / "compound-dtype.nxs"
    dtype = np.dtype([("coordinates", "int16", (2,)), ("weight", "float64")])
    with h5py.File(path, "w") as file:
        file.create_dataset("compound", shape=(3,), dtype=dtype)
    template = create_template(path)
    template.validate().raise_if_errors()
    assert template.retained_datasets()[0].dtype == dtype


def test_retained_reference_payloads_and_attributes_fail_safely(
    tmp_path, no_payload_reads
):
    path = tmp_path / "references.nxs"
    with h5py.File(path, "w") as file:
        target = file.create_dataset("target", shape=(3,), dtype="int64")
        references = file.create_dataset("refs", shape=(1,), dtype=h5py.ref_dtype)
        references.attrs["object_ref"] = target.ref
        file.attrs["region_ref"] = target.regionref[:1]
        file.create_dataset(
            "nested",
            shape=(1,),
            dtype=np.dtype([("value", "i4"), ("ref", h5py.ref_dtype)]),
        )
    template = create_template(path)
    report = template.validate()
    assert any("reference dtype" in d.message for d in report.errors)
    assert any(d.path == "/refs@object_ref" for d in report.errors)
    assert any(d.path == "/@region_ref" for d in report.errors)
    assert any(
        d.path == "/nested" and "reference" in d.message.lower() for d in report.errors
    )
    template["/refs"].set_value([1])
    assert any(d.path == "/refs@object_ref" for d in template.validate().errors)
    assert not any(d.path == "/refs" for d in template.validate().errors)
    del template["/refs"].attrs["object_ref"]
    del template.attrs["region_ref"]
    del template["/nested"]
    template.validate().raise_if_errors()


def test_dynamic_replacement_does_not_reuse_unsafe_reference_dtype(
    tmp_path, no_payload_reads
):
    path = tmp_path / "references.nxs"
    with h5py.File(path, "w") as file:
        file.create_dataset("refs", shape=(1,), dtype=h5py.regionref_dtype)
    template = create_template(path)
    template["/refs"].copy_from("/source/numeric")
    assert "dtype" not in template._snapshot()["/refs"].creation
    assert template.retained_datasets() == []
    template.validate().raise_if_errors()
    assert any(
        "Unsupported inherited dtype" in d.message for d in template.validate().warnings
    )


def test_nonstring_vlen_sizes_are_unknown_and_replacement_is_supported(
    tmp_path, no_payload_reads
):
    path = tmp_path / "vlen.nxs"
    with h5py.File(path, "w") as file:
        file.create_dataset(
            "ragged", shape=(2,), dtype=h5py.vlen_dtype(np.dtype("int32"))
        )
    template = create_template(path)
    assert template.retained_datasets(min_bytes=10**12)[0].logical_bytes is None
    assert any(
        "variable-length output dtypes" in d.message for d in template.validate().errors
    )
    template["/ragged"].copy_from("/source/numeric")
    template.validate().raise_if_errors()
    assert template.retained_datasets() == []


def test_incompatible_inherited_fillvalue_is_reset_and_reported(
    tmp_path, no_payload_reads
):
    path = tmp_path / "fill.nxs"
    with h5py.File(path, "w") as file:
        file.create_dataset("text", shape=(), dtype="S5", fillvalue=b"old")
    template = create_template(path)
    template["/text"].set_value(42)
    assert "fillvalue" not in template._snapshot()["/text"].creation
    report = template.validate()
    report.raise_if_errors()
    assert any(
        "Reset incompatible inherited fillvalue" in d.message for d in report.warnings
    )


def test_virtual_and_external_storage_are_audited_without_opening_targets(
    tmp_path, no_payload_reads, monkeypatch
):
    path = tmp_path / "unsupported.nxs"
    missing = tmp_path / "not-present.h5"
    raw = tmp_path / "not-present.raw"
    with h5py.File(path, "w", libver="latest") as file:
        layout = h5py.VirtualLayout(shape=(5,), dtype="int64")
        layout[:] = h5py.VirtualSource(str(missing), "data", shape=(5,))
        file.create_virtual_dataset("virtual", layout)
        file.create_dataset(
            "external",
            shape=(5,),
            dtype="int64",
            external=[(str(raw), 0, h5py.h5f.UNLIMITED)],
        )
        file["link"] = h5py.ExternalLink(str(missing), "/data")
    real_file = h5py.File
    opened = []

    def guarded_open(filename, mode="r", **kwargs):
        assert filename == path
        assert mode == "r"
        opened.append(filename)
        return real_file(filename, mode, **kwargs)

    monkeypatch.setattr(h5py, "File", guarded_open)
    template = create_template(path)
    records = template.retained_datasets(min_bytes=10**12)
    assert {r.path for r in records} == {"/external", "/virtual"}
    assert all(r.logical_bytes is None and r.storage_bytes is None for r in records)
    report = template.validate()
    assert any("virtual dataset" in d.message for d in report.errors)
    assert any("external raw storage" in d.message for d in report.errors)
    assert any("External dependency" in d.message for d in report.warnings)
    assert opened
    assert not missing.exists()
    assert not raw.exists()
    template["/virtual"].set_value(np.arange(5))
    template["/external"].set_value(np.arange(5))
    template.validate().raise_if_errors()
    assert template.retained_datasets() == []


def test_relative_external_links_block_unsafe_relocation(no_payload_reads):
    template = create_empty_template()
    template.create_link("/external", h5py.ExternalLink("relative.nxs", "/entry"))
    assert any(
        "Relative external links" in d.message for d in template.validate().errors
    )
    del template["/external"]
    template.create_link("/external", h5py.ExternalLink("/absolute.nxs", "/entry"))
    assert not template.validate().errors
    assert template.validate().warnings


def test_move_repairs_absolute_relative_soft_links_and_nexus_target_attributes(
    tmp_path, no_payload_reads
):
    path = tmp_path / "links.nxs"
    with h5py.File(path, "w") as file:
        data = file.create_dataset("old/data", shape=(4,), dtype="int32")
        data.attrs["target"] = "/old/data"
        file.create_dataset("target_array", shape=(0,), dtype="int64").attrs[
            "target"
        ] = [b"/old/data"]
        data.attrs["unrelated"] = "/old/data"
        file.create_group("destination")
        file.create_dataset("outside", shape=(1,), dtype="int32")
        file["absolute"] = h5py.SoftLink("/old/data")
        file["relative"] = h5py.SoftLink("old/data")
        file["old/inside"] = h5py.SoftLink("data")
        file["old/outside"] = h5py.SoftLink("../outside")
    original = path.read_bytes()
    template = create_template(path)
    moved = template.move("/old", "/destination/new", repair_links=True)
    assert moved.path == "/destination/new"
    snapshot = template._snapshot()
    assert "/old" not in snapshot
    assert snapshot["/destination/new/data"].base_path == "/old/data"
    for name in ("/absolute", "/relative", "/destination/new/inside"):
        assert snapshot[name].link.path == "/destination/new/data"
    assert snapshot["/destination/new/outside"].link.path == "/outside"
    attributes = snapshot["/destination/new/data"].attributes
    assert attributes["target"].value == "/destination/new/data"
    assert attributes["unrelated"].value == "/old/data"
    np.testing.assert_array_equal(
        snapshot["/target_array"].attributes["target"].value, ["/destination/new/data"]
    )
    assert any(
        r.path == "/destination/new/data" and r.base_path == "/old/data"
        for r in template.retained_datasets()
    )
    template.validate().raise_if_errors()
    assert "Link repair: /old -> /destination/new" in template.summary()
    assert path.read_bytes() == original


def test_standalone_link_repair_and_unrepaired_move_diagnostics(no_payload_reads):
    template = create_empty_template()
    template.add_dataset("/old", value=1)
    template.create_link("/link", h5py.SoftLink("/old"))
    template.move("/old", "/new")
    assert any("without link repair" in d.message for d in template.validate().warnings)
    assert any(
        "Unresolved soft link" in d.message for d in template.validate().warnings
    )
    assert template.repair_links("/old", "/new") is template
    assert template._snapshot()["/link"].link.path == "/new"
    assert not any(
        "Unresolved soft link" in d.message for d in template.validate().warnings
    )


def test_deleting_a_nexus_target_is_diagnosed_without_rewriting_arbitrary_attrs(
    no_payload_reads,
):
    template = create_empty_template()
    template.add_dataset("/data", value=1)
    metadata = template.add_group("/metadata")
    metadata.attrs["target"] = "/data"
    metadata.attrs["unrelated"] = "/data"
    del template["/data"]
    assert any(
        d.path == "/metadata@target" and "Unresolved NeXus target" in d.message
        for d in template.validate().warnings
    )
    assert template._snapshot()["/metadata"].attributes["unrelated"].value == "/data"


def test_hard_group_aliases_share_descendants_and_audit_identity(
    base_file, no_payload_reads
):
    template = create_template(base_file)
    template.create_link("/entry_alias", "/entry")
    template["/entry_alias"].add_dataset("added", value=1)
    snapshot = template._snapshot()
    assert snapshot["/entry"].object_key == snapshot["/entry_alias"].object_key
    assert (
        snapshot["/entry/added"].object_key == snapshot["/entry_alias/added"].object_key
    )
    record = next(r for r in template.retained_datasets() if r.path == "/entry/data")
    assert record.aliases == ("/entry_alias/data", "/other/alias")
    assert len(template.retained_datasets()) == 4
    assert any("aliases" in d.message for d in template.validate().warnings)
    with pytest.raises(ValueError, match="cycle"):
        template.create_link("/entry/cycle", "/entry")
    with pytest.raises(ValueError, match="descendant"):
        template.move("/entry", "/entry_alias/child")


def test_existing_group_cycles_terminate_and_block_validation(
    tmp_path, no_payload_reads
):
    path = tmp_path / "cycle.nxs"
    with h5py.File(path, "w") as file:
        group = file.create_group("group")
        group["cycle"] = group
    template = create_template(path)
    assert set(template._snapshot()) == {"/", "/group", "/group/cycle"}
    assert any(
        "Cyclic hard-linked groups" in d.message for d in template.validate().errors
    )
    assert len(template.summary()) < 3000


def test_unresolvable_relative_links_in_moves_are_explicitly_blocked(no_payload_reads):
    template = create_empty_template()
    template.add_group("/old")
    template.add_group("/destination")
    template.create_link("/old/broken", h5py.SoftLink("../missing"))
    template.move("/old", "/destination/new", repair_links=True)
    assert any("Cannot safely repair" in d.message for d in template.validate().errors)
    del template["/destination/new/broken"]
    template.validate().raise_if_errors()


def test_relative_soft_link_repairs_do_not_silently_change_group_alias_targets(
    no_payload_reads,
):
    template = create_empty_template()
    template.add_group("/old")
    template.add_group("/deep")
    template.add_dataset("/outside", value=1)
    template.add_dataset("/deep/outside", value=2)
    template.create_link("/old/link", h5py.SoftLink("../outside"))
    template.create_link("/deep/alias", "/old")
    template.move("/old", "/deep/new", repair_links=True)
    assert any(
        "ambiguous through shared group aliases" in d.message
        for d in template.validate().errors
    )
    del template["/deep/new/link"]
    template.create_link("/deep/new/link", h5py.SoftLink("/outside"))
    template.validate().raise_if_errors()


def test_unsupported_target_array_repairs_can_be_resolved_explicitly(no_payload_reads):
    template = create_empty_template()
    template.add_dataset("/old", value=1)
    template.add_group("/metadata").attrs["target"] = np.array(["/old", "/old"])
    template.move("/old", "/new", repair_links=True)
    assert any(
        "Only scalar or length-one" in d.message for d in template.validate().errors
    )
    template["/metadata"].attrs["target"] = "/new"
    template.validate().raise_if_errors()


def test_moves_through_shared_parent_group_aliases_are_explicitly_blocked(
    no_payload_reads,
):
    template = create_empty_template()
    template.add_group("/parent")
    template.add_group("/destination")
    template.add_dataset("/parent/value", value=1)
    template.create_link("/alias", "/parent")
    template.move("/parent/value", "/destination/value", repair_links=True)
    assert any(
        "shared parent-group aliases is unsupported" in d.message
        for d in template.validate().errors
    )


@pytest.mark.parametrize("path", ["relative", "/bad/../path", "/bad\0path"])
def test_invalid_output_paths_are_rejected(path):
    with pytest.raises(ValueError, match="root-absolute"):
        create_empty_template().add_group(path)


def test_validation_report_contract_and_missing_base(tmp_path):
    error = Diagnostic("error", "/entry", "not valid")
    warning = Diagnostic("warning", "/other", "advisory")
    report = ValidationReport((error, warning))
    assert report.errors == (error,)
    assert report.warnings == (warning,)
    with pytest.raises(ValueError, match="/entry: not valid"):
        report.raise_if_errors()
    ValidationReport((warning,)).raise_if_errors()
    template = create_template(tmp_path / "missing.nxs")
    assert any(
        "Cannot inspect template metadata" in d.message
        for d in template.validate().errors
    )
    assert "error /:" in template.summary()


@pytest.mark.parametrize("value", [-1, True, 1.2])
def test_invalid_thresholds_and_limits_are_rejected(value):
    template = create_empty_template()
    with pytest.raises(ValueError, match="min_bytes"):
        template.retained_datasets(min_bytes=value)
    with pytest.raises(ValueError, match="retained_warning_bytes"):
        template.validate(retained_warning_bytes=value)
    with pytest.raises(ValueError, match="limit"):
        template.summary(limit=value)


def test_creation_and_binding_input_errors():
    template = create_empty_template()
    with pytest.raises(TypeError, match="unknown dataset creation"):
        template.add_dataset("/invalid", value=1, made_up=True)
    assert "/invalid" not in template
    target = template.add_dataset("/value", shape=(1,), dtype="int64")
    with pytest.raises(TypeError, match="callable"):
        target.set_from(42)
    with pytest.raises(TypeError, match="field name"):
        target.set_from_result(SourceSpec(), None)
    with pytest.raises(ValueError, match="root-absolute"):
        target.copy_from("relative/path")
    with pytest.raises(TypeError, match="provider specification"):
        target.copy_from(np.arange(3))
    with pytest.raises(TypeError, match="not a string"):
        target.copy_from(SourceSpec(), columns="id")
    target.copy_from("/entry/value", read_chunk_size=0)
    assert any("read_chunk_size" in d.message for d in template.validate().errors)
    assert Binding("literal", 1).options is not Binding("literal", 2).options


def test_live_hdf5_objects_cannot_be_used_as_lazy_literals(base_file, no_payload_reads):
    template = create_empty_template()
    with h5py.File(base_file, "r") as file:
        with pytest.raises(TypeError, match="live HDF5 objects"):
            template.add_dataset("/bad", value=file["/entry/data"])
        target = template.add_dataset("/target", value=1)
        with pytest.raises(TypeError, match="live HDF5 objects"):
            target.set_value(file["/entry/data"])
        with pytest.raises(TypeError, match="live HDF5 objects"):
            target.set_value(file["/entry/data"], selection=())
        with pytest.raises(TypeError, match="live HDF5 objects"):
            target.attrs["bad"] = file["/entry/data"]
        with pytest.raises(TypeError, match="live HDF5 objects"):
            template.add_dataset(
                "/bad_fill", shape=(1,), dtype="int64", fillvalue=file["/entry/data"]
            )
        template.validate().raise_if_errors()


def test_unsupported_named_datatypes_are_diagnosed_and_can_be_removed(
    tmp_path, no_payload_reads
):
    path = tmp_path / "named-type.nxs"
    with h5py.File(path, "w") as file:
        file["named_type"] = np.dtype("int32")
    template = create_template(path)
    assert any(
        "Unsupported HDF5 object Datatype at /named_type" in d.message
        for d in template.validate().errors
    )
    del template["/named_type"]
    template.validate().raise_if_errors()
