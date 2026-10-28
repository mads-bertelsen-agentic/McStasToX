# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Mccode-dev contributors (https://github.com/mccode-dev)
"""Synthetic-HDF5 coverage of source specifications and column transforms."""

from dataclasses import FrozenInstanceError

import h5py
import numpy as np
import pytest

from mcstastox.LoadFile import Data
from mcstastox.NexusSource import (
    ChunkOperation,
    ColumnOperation,
    EventColumn,
    EventSource,
    SourceContext,
    event_source,
)
from mcstastox.Sampling import SamplingSettings, sample_event_chunks

_EVENTS = np.array(
    [
        [0, 0.000, 0, 1, 1.0],
        [1, 0.001, 1, 0, 1.5],
        [2, 0.002, 2, 2, 2.0],
        [0, 0.003, 3, 3, 2.5],
        [4, 0.004, 4, 4, 3.0],
    ]
)


def _write_source(path, *, components=None, entries=(1, 2, 10)):
    second = _EVENTS.copy()
    second[:, 2] += 10
    if components is None:
        components = [
            ("First", _EVENTS, "p t id w L", True),
            ("Second", second, "p t id w L", True),
            ("Empty", np.empty((0, 5)), "p t id w L", True),
            ("NoPixels", _EVENTS, "p t id w L", False),
        ]
    with h5py.File(path, "w") as handle:
        for number in entries:
            entry = handle.create_group(f"entry{number}")
            entry.create_group("data")
            simulation = entry.create_group("simulation")
            simulation.attrs["program"] = np.bytes_("3.8.6")
            simulation.create_group("Param")
            entry.create_dataset("title", data=np.bytes_(f"00{number}"))
            entry.create_dataset(
                "text", data=["1.25", "caf\u00e9"], dtype=h5py.string_dtype("utf-8")
            )
            entry.create_dataset("scalar", data=1.25)
            entry.create_dataset(
                "matrix", data=np.arange(24, dtype=np.int16).reshape(4, 6)
            )
            entry.create_dataset("empty", shape=(0, 3), dtype="int32")
            component_group = entry.create_group("instrument/components")
            for index, (name, events, variables, pixels) in enumerate(components):
                output = component_group.create_group(f"{index:04d}_{name}/output")
                if pixels:
                    output.create_group("BINS")
                info = output.create_group("event_data")
                info.attrs["variables"] = np.bytes_(variables)
                info.create_dataset("events", data=events)
    return path


@pytest.fixture
def source_path(tmp_path):
    return _write_source(tmp_path / "source.nxs")


def _no_payload(*args, **kwargs):
    raise AssertionError("Dataset payload must not be read")


def test_path_context_owns_handle_and_selects_sparse_entries(source_path):
    with SourceContext(source_path, entry_number=10) as context:
        raw = context.raw
        assert isinstance(raw, h5py.File)
        assert context.mcstas.file_handle is raw
        assert context.entry_name == "entry10"
        assert isinstance(context.entry.raw, h5py.Group)
        assert context.entry.raw.name == "/entry10"
        assert context.entry.read_text("title") == "0010"
        assert context.read_text("/entry1/title") == "001"
        assert raw.id.valid
    assert not raw.id.valid
    with pytest.raises(ValueError, match="closed"):
        context.read("/entry1/title")
    with pytest.raises(ValueError, match="closed"):
        context.__enter__()


def test_owned_handle_closes_on_failure(source_path):
    context = SourceContext(str(source_path))
    raw = context.raw
    assert context.entry_name == "entry1"
    with pytest.raises(RuntimeError, match="failure"), context:
        raise RuntimeError("failure")
    assert not raw.id.valid


def test_borrowed_data_and_context_never_close_or_switch_entry(source_path):
    with Data(source_path.parent, filename=source_path.name, entry_number=2) as data:
        with SourceContext(data, entry_number=2) as context:
            assert context.raw is data.file
            assert context.mcstas is data.file_object
            with SourceContext(context, entry_number=2) as borrowed:
                assert borrowed.raw is context.raw
            assert context.raw.id.valid
            with pytest.raises(ValueError, match="conflicts"):
                SourceContext(context, entry_number=10)
        assert data.file.id.valid
        with pytest.raises(ValueError, match="conflicts"):
            SourceContext(data, entry_number=1)
        with pytest.raises(RuntimeError):
            with SourceContext(data):
                raise RuntimeError("failure")
        assert data.file.id.valid


def test_borrowing_owned_context_preserves_owner_lifetime(source_path):
    with SourceContext(source_path) as owner:
        borrowed = SourceContext(owner)
        borrowed.close()
        assert owner.raw.id.valid
    with pytest.raises(ValueError, match="closed"):
        borrowed.raw
    with pytest.raises(ValueError, match="closed"):
        SourceContext(owner)


def test_closed_data_is_rejected(source_path):
    data = Data(source_path.parent, filename=source_path.name)
    data.close()
    with pytest.raises(ValueError, match="closed"):
        SourceContext(data)


@pytest.mark.parametrize("entry_number", [0, -1, True, 1.0, "2"])
def test_entry_override_validation(source_path, entry_number):
    with pytest.raises((TypeError, ValueError), match="entry_number"):
        SourceContext(source_path, entry_number=entry_number)


def test_missing_entry_rejected(source_path):
    with pytest.raises(ValueError, match="entry3"):
        SourceContext(source_path, entry_number=3)


def test_none_context_is_lazy_and_can_be_borrowed():
    with SourceContext(None) as context, SourceContext(context) as borrowed:
        assert event_source(variables="id").description
        for get in (
            lambda: context.raw,
            lambda: context.mcstas,
            lambda: context.entry_name,
            lambda: context.entry.raw,
            lambda: borrowed.entry.read("title"),
            lambda: event_source().schema(context),
        ):
            with pytest.raises(ValueError, match="source is None"):
                get()
    with pytest.raises(TypeError, match="source"):
        SourceContext(object())


def test_raw_and_decoded_reads_preserve_dtype_text_and_selected_shape(source_path):
    with SourceContext(source_path, entry_number=2) as context:
        assert context.entry.read("title") == b"002"
        assert context.entry.read_text("title") == "002"
        assert isinstance(context.entry.read_text("title"), str)
        text = context.entry.read_text("text")
        assert text.shape == (2,)
        assert text.tolist() == ["1.25", "caf\u00e9"]
        assert context.entry.read_text("text", selection=0) == "1.25"
        matrix = context.entry.read("matrix")
        assert matrix.dtype == np.dtype("int16")
        assert context.entry.read("scalar").shape == ()
        assert context.entry.read("matrix", selection=1).shape == (6,)
        np.testing.assert_array_equal(
            context.read("/entry1/matrix", selection=(slice(1, 3), slice(2, 5))),
            matrix[1:3, 2:5],
        )
        with pytest.raises(TypeError, match="text dataset"):
            context.entry.read_text("scalar")
        with pytest.raises(TypeError, match="not a dataset"):
            context.read("/entry1")
        with pytest.raises(ValueError, match="absolute"):
            context.read("entry1/title")
        with pytest.raises(ValueError, match="relative"):
            context.entry.read("/entry1/title")
        with pytest.raises(ValueError, match="cannot contain"):
            context.entry.read("../entry1/title")


@pytest.mark.parametrize("chunk_size", [1, 3, 100])
@pytest.mark.parametrize(
    "selection",
    [(), (slice(1, 4, 2), slice(1, 6, 2)), (2, slice(1, 6, 2)), (Ellipsis, 2)],
)
def test_dataset_iteration_matches_basic_selection(source_path, chunk_size, selection):
    with SourceContext(source_path) as context:
        chunks = list(
            context.entry.iter_dataset(
                "matrix", read_chunk_size=chunk_size, selection=selection
            )
        )
        expected = context.entry.read("matrix", selection=selection)
        np.testing.assert_array_equal(np.concatenate(chunks), expected)
        assert all(len(chunk) <= chunk_size for chunk in chunks)


def test_dataset_iteration_scalar_empty_and_invalid_selections(source_path):
    with SourceContext(source_path) as context:
        assert list(context.entry.iter_dataset("scalar")) == [1.25]
        assert list(context.entry.iter_dataset("matrix", selection=(-1, 2))) == [20]
        assert list(context.entry.iter_dataset("empty")) == []
        assert list(context.entry.iter_dataset("matrix", selection=slice(3, 1))) == []
        for selection in (
            [0, 1],
            (Ellipsis, Ellipsis),
            (1, 2, 3),
            4,
            slice(None, None, -1),
        ):
            with pytest.raises((ValueError, TypeError, IndexError)):
                list(context.entry.iter_dataset("matrix", selection=selection))


@pytest.mark.parametrize("chunk_size", [None, 0, -1, True, 1.5, "2"])
def test_dataset_iteration_validates_chunk_size(source_path, chunk_size):
    with SourceContext(source_path) as context:
        with pytest.raises((TypeError, ValueError), match="chunk_size"):
            list(context.entry.iter_dataset("matrix", read_chunk_size=chunk_size))


def test_schema_and_counts_are_metadata_only_even_for_empty_tables(
    source_path, monkeypatch
):
    with SourceContext(source_path) as context:
        monkeypatch.setattr(h5py.Dataset, "__getitem__", _no_payload)
        monkeypatch.setattr(context.mcstas, "iter_event_data", _no_payload)
        events = event_source(variables=("id", "t")).sample(
            SamplingSettings(7, weight_key="w")
        )
        assert events.schema(context) == dict.fromkeys(
            ("id", "t", "w"), np.dtype("float64")
        )
        assert events.row_count(context) == 10
        empty = event_source(component_name="Empty", variables="id")
        assert empty.schema(context) == {"id": np.dtype("float64")}
        assert empty.row_count(context) == 0
        filtered = event_source(variables="id", filter_zeros=True, weight_key="w")
        assert filtered.row_count(context) is None
        assert list(filtered.schema(context)) == ["id", "w"]


@pytest.mark.parametrize("trailing_spaces", [" ", "   "])
def test_monitor_nd_trailing_spaces_preserve_schema_and_event_columns(
    tmp_path, monkeypatch, trailing_spaces
):
    table = np.array(
        [
            [0, 0.1, -0.2, 11, 101, 0.001, 2.5],
            [2, 0.3, -0.4, 12, 102, 0.002, 3.5],
            [4, 0.5, -0.6, 13, 103, 0.003, 4.5],
        ]
    )
    path = _write_source(
        tmp_path / "source.nxs",
        components=[("First", table, "p x y n id t L" + trailing_spaces, True)],
        entries=(1,),
    )
    events = event_source(
        variables=("id", "t", "L", "x", "y", "n"), read_chunk_size=2
    ).sample(SamplingSettings(7))
    assert "p" not in events.variables
    assert events.required_variables == ("id", "t", "L", "x", "y", "n", "p")
    with SourceContext(path) as context:
        with monkeypatch.context() as metadata_only:
            metadata_only.setattr(h5py.Dataset, "__getitem__", _no_payload)
            metadata_only.setattr(context.mcstas, "iter_event_data", _no_payload)
            assert events.schema(context) == dict.fromkeys(
                events.required_variables, np.dtype("float64")
            )
            assert events.row_count(context) == len(table)
        chunks = list(events.iter_chunks(context))
    assert [len(chunk["id"]) for chunk in chunks] == [2, 1]
    assert all(set(chunk) == set(events.required_variables) for chunk in chunks)
    for index, variable in enumerate(("p", "x", "y", "n", "id", "t", "L")):
        np.testing.assert_array_equal(
            np.concatenate([chunk[variable] for chunk in chunks]), table[:, index]
        )


def test_empty_component_needs_no_first_batch(source_path, monkeypatch):
    with SourceContext(source_path) as context:
        monkeypatch.setattr(h5py.Dataset, "__getitem__", _no_payload)
        events = event_source(component_name="Empty", variables=("id", "t"))
        assert list(events.iter_chunks(context)) == []
        assert events["id"].scale(2).output_dtype(
            events.schema(context)["id"]
        ) == np.dtype("float64")


@pytest.mark.parametrize(
    "component_name",
    [None, ["Second", "First"], ("Second", "Empty", "First"), "NoPixels"],
)
def test_components_and_rows_retain_explicit_or_default_order(
    source_path, component_name
):
    events = event_source(
        component_name=component_name, variables=("id", "t", "p"), read_chunk_size=2
    )
    with SourceContext(source_path) as context:
        chunks = list(events.iter_chunks(context))
    first = np.arange(5)
    expected = (
        first
        if component_name == "NoPixels"
        else np.concatenate(
            (first, first + 10) if component_name is None else (first + 10, first)
        )
    )
    np.testing.assert_array_equal(
        np.concatenate([chunk["id"] for chunk in chunks]), expected
    )
    assert all(0 < len(chunk["id"]) <= 2 for chunk in chunks)
    assert sum(np.count_nonzero(chunk["p"] == 0) for chunk in chunks) == (
        2 if component_name == "NoPixels" else 4
    )
    for chunk in chunks:
        np.testing.assert_allclose(chunk["t"], chunk["id"] % 10 * 0.001)


def test_source_specs_are_normalized_immutable_and_identity_based():
    names, variables = ["Second", "First"], ["id", "t"]
    original = event_source(
        component_name=names, variables=variables, read_chunk_size=17
    )
    names.reverse()
    variables.append("p")
    assert original.component_name == ("Second", "First")
    assert original.variables == ("id", "t")
    settings = SamplingSettings(9, seed=42, ordered=True, weight_key="w")
    sampled, independent = original.sample(settings), original.sample(settings)
    assert len({original, sampled, independent}) == 3
    assert sampled != independent
    assert sampled.sampling is settings
    assert original.sampling is None
    assert sampled.read_chunk_size == 17
    assert sampled.component_name == original.component_name
    assert sampled.required_variables == ("id", "t", "w")
    assert "requested=9" in sampled.description
    assert "weight_key='w'" in sampled.description
    assert "ordered=True" in sampled.description
    with pytest.raises(FrozenInstanceError):
        original.filter_zeros = True
    with pytest.raises(TypeError, match="SamplingSettings"):
        original.sample(9)
    with pytest.raises(KeyError, match="not declared"):
        sampled["w"]


def test_preprocessing_contract_separates_raw_and_processed_schemas(
    tmp_path, monkeypatch
):
    path = _write_source(
        tmp_path / "logger.nxs",
        components=[("Logger", _EVENTS, "p t x w L", False)],
    )
    schema = {"p": "float64", "id": "int64", "t": "float64"}
    raw = event_source(component_name="Logger", variables=("x", "t"))
    prepared = raw.preprocess(
        _no_payload, schema=schema, description="Voxel acceptance"
    )
    schema["id"] = "float32"
    sampled = prepared.sample(SamplingSettings(7, seed=42))
    assert raw.output_variables == ("x", "t")
    assert prepared.variables == raw.variables
    assert sampled.output_variables == ("p", "id", "t")
    assert sampled.required_variables == ("x", "t", "p")
    assert sampled["id"].variable == "id"
    with pytest.raises(KeyError, match="not declared"):
        sampled["x"]
    with pytest.raises(ValueError, match="before sample"):
        sampled.preprocess(_no_payload, schema=schema)
    with pytest.raises(TypeError):
        sampled.preprocessors[0].schema["id"] = np.dtype("float32")
    assert isinstance(prepared.preprocessors[0], ChunkOperation)
    assert "Voxel acceptance" in prepared.description
    with SourceContext(path, entry_number=2) as context:
        monkeypatch.setattr(h5py.Dataset, "__getitem__", _no_payload)
        assert sampled.raw_schema(context) == dict.fromkeys(
            ("x", "t", "p"), np.dtype("float64")
        )
        assert sampled.schema(context) == {
            "p": np.dtype("float64"),
            "id": np.dtype("int64"),
            "t": np.dtype("float64"),
        }
        assert sampled.row_count(context) is None


def test_preprocessing_chain_checks_custom_hidden_weight_when_sampling():
    raw = event_source(variables=("id", "t"), filter_zeros=True, weight_key="p")
    first = raw.preprocess(_no_payload, schema={"id": "int64", "w": "float64"})
    second = first.preprocess(_no_payload, schema={"pixel": "int64", "w": "float64"})
    sampled = second.sample(SamplingSettings(3, weight_key="w"))
    assert sampled.required_variables == ("id", "t", "w", "p")
    assert sampled.output_variables == ("pixel", "w")
    assert len(sampled.preprocessors) == 2
    with pytest.raises(ValueError, match="retain sampling weight 'p'"):
        second.sample(SamplingSettings(3))
    with pytest.raises(ValueError, match="retain sampling weight 'w'"):
        raw.preprocess(_no_payload, schema={"id": "int64"}).preprocess(
            _no_payload, schema={"id": "int64", "w": "float64"}
        ).sample(SamplingSettings(3, weight_key="w"))
    with pytest.raises(TypeError, match="weights must be real"):
        raw.preprocess(_no_payload, schema={"p": "complex128"}).sample(
            SamplingSettings(3)
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"callback": None, "schema": {"id": "int64"}},
        {"callback": _no_payload, "schema": []},
        {"callback": _no_payload, "schema": {}},
        {"callback": _no_payload, "schema": {"bad name": "int64"}},
        {"callback": _no_payload, "schema": {1: "int64"}},
        {"callback": _no_payload, "schema": {"id": "S4"}},
        {"callback": _no_payload, "schema": {"id": "int64"}, "description": 1},
    ],
)
def test_preprocessing_specification_rejects_invalid_contracts(kwargs):
    with pytest.raises((ValueError, TypeError)):
        event_source().preprocess(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"variables": []},
        {"variables": ["id", "id"]},
        {"variables": ["t id"]},
        {"variables": {"id"}},
        {"component_name": []},
        {"component_name": ["First", "First"]},
        {"filter_zeros": 1},
        {"weight_key": ""},
        {"weight_key": ["p"]},
        {"read_chunk_size": None},
        {"read_chunk_size": True},
        {"read_chunk_size": 0},
    ],
)
def test_invalid_source_configuration(kwargs):
    with pytest.raises((TypeError, ValueError)):
        event_source(**kwargs)


@pytest.mark.parametrize(
    ("variables", "requested"),
    [
        ("pp t id w L", "p"),
        ("p t identity w L", "id"),
        ("p t id w L", "missing"),
    ],
)
def test_schema_uses_exact_variable_tokens(tmp_path, variables, requested):
    path = _write_source(
        tmp_path / "source.nxs", components=[("First", _EVENTS, variables, True)]
    )
    with SourceContext(path) as context:
        with pytest.raises(ValueError, match=f"exact event variable '{requested}'"):
            event_source(variables=requested).schema(context)
        with pytest.raises(ValueError, match="exact event variable"):
            list(event_source(variables=requested).iter_chunks(context))


@pytest.mark.parametrize(
    ("events", "variables", "error"),
    [
        (np.empty(5), "p t id w L", "2D table"),
        (np.empty((0, 4)), "p t id w L", "5 columns"),
        (np.empty((0, 4)), "p t id w L   ", "5 columns"),
        (np.empty((0, 5), dtype="S1"), "p t id w L", "numeric dtype"),
        (np.empty((0, 5), dtype="bool"), "p t id w L", "numeric dtype"),
        (_EVENTS, "p t id id L", "duplicate"),
        (_EVENTS, "p  t id w L", "space-separated"),
        (_EVENTS, "p  t id w L ", "space-separated"),
        (_EVENTS, " p t id w L ", "space-separated"),
        (_EVENTS, "p\tt id w L ", "space-separated"),
        (_EVENTS, "p t id w L\t ", "space-separated"),
    ],
)
def test_bad_event_metadata_rejected_without_payload_reads(
    tmp_path, monkeypatch, events, variables, error
):
    path = _write_source(
        tmp_path / "source.nxs", components=[("First", events, variables, True)]
    )
    with SourceContext(path) as context:
        monkeypatch.setattr(h5py.Dataset, "__getitem__", _no_payload)
        with pytest.raises((TypeError, ValueError), match=error):
            event_source(variables="id").schema(context)


def test_schema_checks_later_components_before_any_payload(tmp_path, monkeypatch):
    path = _write_source(
        tmp_path / "source.nxs",
        components=[
            ("First", _EVENTS, "p t id w L", True),
            ("Second", _EVENTS.astype("float32"), "p t id w L", True),
        ],
    )
    with SourceContext(path) as context:
        monkeypatch.setattr(h5py.Dataset, "__getitem__", _no_payload)
        with pytest.raises(ValueError, match=r"Second.*differs"):
            list(event_source(variables="id").iter_chunks(context))


def test_missing_hidden_weights_and_no_default_components(tmp_path, source_path):
    with SourceContext(source_path) as context:
        events = event_source(variables="id").sample(
            SamplingSettings(2, weight_key="absent")
        )
        with pytest.raises(ValueError, match="absent"):
            events.schema(context)
    path = _write_source(
        tmp_path / "no_pixels.nxs", components=[("Only", _EVENTS, "p t id w L", False)]
    )
    with SourceContext(path) as context:
        with pytest.raises(ValueError, match="No event components"):
            event_source().schema(context)
        with pytest.raises(ValueError, match="No component"):
            event_source(component_name="absent").schema(context)


@pytest.mark.parametrize("ordered", [False, True])
@pytest.mark.parametrize("filter_zeros", [False, True])
def test_hidden_custom_weights_reuse_existing_sampler_exactly(
    source_path, ordered, filter_zeros
):
    settings = SamplingSettings(19, seed=42, ordered=ordered, weight_key="w")
    events = event_source(
        component_name=("Second", "First"),
        variables=("id", "t"),
        read_chunk_size=2,
        filter_zeros=filter_zeros,
        weight_key="p",
    ).sample(settings)
    assert events.required_variables == (
        ("id", "t", "w", "p") if filter_zeros else ("id", "t", "w")
    )
    with SourceContext(source_path) as context:
        chunks = list(events.iter_chunks(context))
        assert sum(len(chunk["id"]) for chunk in chunks) == (6 if filter_zeros else 10)
        sampled = sample_event_chunks(chunks, settings)
        direct_chunks = context.mcstas.iter_event_data(
            events.required_variables, component_name=["Second", "First"], chunk_size=2
        )
        if filter_zeros:
            direct_chunks = (
                {key: value[chunk["p"] != 0] for key, value in chunk.items()}
                for chunk in direct_chunks
            )
        direct = sample_event_chunks(direct_chunks, settings)
    for key in events.required_variables:
        np.testing.assert_array_equal(sampled[key], direct[key])
    np.testing.assert_array_equal(sampled["w"], np.ones(19))
    assert sampled.total_weight == direct.total_weight
    assert sampled.total_weight_variance == direct.total_weight_variance
    assert sampled.effective_duration == direct.effective_duration
    np.testing.assert_allclose(sampled["t"], sampled["id"] % 10 * 0.001)


def test_explicit_filter_weight_is_hidden_and_independent_of_sampling_weight(
    source_path,
):
    events = event_source(
        component_name="First", variables="id", filter_zeros=True, weight_key="w"
    )
    with SourceContext(source_path) as context:
        chunks = list(events.iter_chunks(context))
        np.testing.assert_array_equal(chunks[0]["id"], [0, 2, 3, 4])
        assert set(chunks[0]) == {"id", "w"}
        assert events.row_count(context) is None


def test_column_pipeline_is_immutable_structured_and_ordered():
    events = event_source(variables="id")
    original = events["id"]
    column = original.scale(2).offset(3).convert_units("m", "cm", dtype="float64")
    assert isinstance(events, EventSource)
    assert isinstance(column, EventColumn)
    assert column.source is events
    assert original.operations == ()
    assert isinstance(column.operations, tuple)
    assert all(
        isinstance(operation, ColumnOperation) for operation in column.operations
    )
    assert [operation.kind for operation in column.operations] == [
        "scale",
        "offset",
        "convert_units",
    ]
    assert column.operations[0].parameters["factor"] == 2
    assert column.output_dtype("int16") == np.dtype("float64")
    np.testing.assert_array_equal(column.apply({"id": np.arange(3)}), [300, 500, 700])
    assert (
        column.description.index("scale by")
        < column.description.index("offset by")
        < column.description.index("convert m to cm")
    )
    assert column.target_unit == "cm"
    with pytest.raises(FrozenInstanceError):
        column.variable = "t"
    with pytest.raises(TypeError):
        column.operations[0].parameters["factor"] = 10


@pytest.mark.parametrize(
    ("source_unit", "target_unit", "factor"),
    [
        ("s", "us", 1_000_000),
        ("ms", "us", 1000),
        ("ns", "s", 1e-9),
        ("m", "mm", 1000),
        ("nm", "A", 10),
        ("angstrom", "m", 1e-10),
    ],
)
def test_standard_unit_registry(source_unit, target_unit, factor):
    column = event_source(variables="t")["t"].convert_units(source_unit, target_unit)
    assert column.target_unit == target_unit
    assert column.operations[0].parameters["factor"] == factor
    assert column.output_dtype("float32") == np.dtype("float64")
    np.testing.assert_allclose(
        column.apply({"t": np.array([1.0, 2.0])}), [factor, 2 * factor]
    )


def test_unit_mismatch_unknown_units_and_chained_declarations():
    column = event_source(variables="t")["t"]
    with pytest.raises(ValueError, match="dimension mismatch"):
        column.convert_units("s", "m")
    with pytest.raises(ValueError, match="Unsupported"):
        column.convert_units("s", "fortnight")
    converted = column.convert_units("s", "us").convert_units("us", "ms")
    assert converted.target_unit == "ms"
    np.testing.assert_allclose(converted.apply({"t": np.array([1.0])}), [1000.0])
    with pytest.raises(ValueError, match="Previous conversion"):
        converted.convert_units("s", "ns")
    assert converted.map(lambda values: values, dtype="float64").target_unit is None


def test_mapping_snapshots_exact_keys_and_preserves_duplicates_and_empty_schema():
    original = {0: 1001, 1: 1002, 2: 1010}
    column = event_source(variables="id")["id"].map_values(original, dtype="int64")
    original[0] = -99
    np.testing.assert_array_equal(
        column.apply({"id": np.array([2.0, 0, 2, 1])}), [1010, 1001, 1010, 1002]
    )
    assert column.output_dtype("float64") == np.dtype("int64")
    empty = column.apply({"id": np.empty(0)})
    assert empty.shape == (0,)
    assert empty.dtype == np.dtype("int64")
    assert "3 entries" in column.description
    assert "1001" not in column.description
    with pytest.raises(TypeError):
        column.operations[0].parameters["mapping"][0] = 0
    for values in ([1.5], [99]):
        with pytest.raises(ValueError, match=r"Missing mapping.*'id'"):
            column.apply({"id": np.array(values)})


def test_mapping_keep_still_checks_integer_integrality_and_range():
    column = event_source(variables="id")["id"].map_values(
        {1: 10}, dtype="int8", missing="keep"
    )
    np.testing.assert_array_equal(column.apply({"id": np.array([1.0, 2])}), [10, 2])
    with pytest.raises(ValueError, match="integral"):
        column.apply({"id": np.array([1.5])})
    with pytest.raises(OverflowError, match="range"):
        column.apply({"id": np.array([128])})


@pytest.mark.parametrize(
    ("mapping", "dtype", "error"),
    [
        ({0: 1.5}, "int64", ValueError),
        ({0: 128}, "int8", OverflowError),
        ({0: -1}, "uint64", OverflowError),
        ({0: float(2**63)}, "int64", OverflowError),
        ({0: 2**64}, "uint64", OverflowError),
        ({0: np.nan}, "int64", ValueError),
        ({0: np.inf}, "int64", ValueError),
        ({0: "text"}, "int64", TypeError),
        ({0: [1, 2]}, "int64", ValueError),
    ],
)
def test_mapping_rejects_unsafe_declared_values(mapping, dtype, error):
    with pytest.raises(error):
        event_source(variables="id")["id"].map_values(mapping, dtype=dtype)


def test_schema_descriptions_and_reprs_never_execute_callbacks(
    source_path, monkeypatch
):
    calls = []

    def correct_time(values):
        calls.append(len(values))
        return values + 1

    column = (
        event_source(component_name="First", variables="t")["t"]
        .map(correct_time, dtype="float64", description="User time correction")
        .scale(2)
    )
    with SourceContext(source_path) as context:
        monkeypatch.setattr(h5py.Dataset, "__getitem__", _no_payload)
        schema = column.source.schema(context)
        assert column.output_dtype(schema["t"]) == np.dtype("float64")
        assert (
            "custom callback correct_time (User time correction)" in column.description
        )
        assert "custom callback" in repr(column)
        assert "First" in repr(column.source)
        assert calls == []
    np.testing.assert_array_equal(column.apply({"t": np.array([1, 2])}), [4, 6])
    assert calls == [2]
    anonymous = event_source(variables="id")["id"].map(
        lambda value: value, dtype="int64"
    )
    assert "custom callback" in anonymous.description
    assert "lambda" not in anonymous.description


def test_callback_cannot_mutate_sibling_projection_inputs():
    def mutate(values):
        values[:] = 42
        return values

    events = event_source(variables=("id", "t"))
    chunk = {"id": np.array([0.0, 1]), "t": np.array([3.0, 4])}
    np.testing.assert_array_equal(
        events["id"].map(mutate, dtype="int64").apply(chunk), [42, 42]
    )
    np.testing.assert_array_equal(events["id"].apply(chunk), [0, 1])


@pytest.mark.parametrize("output", [1, [1], [[1, 2]], [1, 2, 3]])
def test_callback_must_preserve_one_dimensional_row_count(output):
    column = event_source(variables="id")["id"].map(
        lambda values: output, dtype="float64"
    )
    with pytest.raises(ValueError, match="one-dimensional row count"):
        column.apply({"id": np.array([1.0, 2])})


@pytest.mark.parametrize(
    "chunk",
    [
        {"id": np.zeros((2, 1))},
        {"id": np.ones(2), "t": np.ones(3)},
        {"id": np.array(1)},
    ],
)
def test_column_rejects_misaligned_or_non_vector_input(chunk):
    with pytest.raises(ValueError, match=r"one-dimensional|same row count"):
        event_source(variables="id")["id"].apply(chunk)


def test_integer_arithmetic_promotes_small_types_and_checks_overflow():
    column = event_source(variables="id")["id"]
    assert column.scale(2).output_dtype("int8") == np.dtype("int64")
    np.testing.assert_array_equal(
        column.scale(2).apply({"id": np.array([127], dtype="int8")}), [254]
    )
    np.testing.assert_array_equal(
        column.offset(-1).apply({"id": np.array([0], dtype="uint8")}), [-1]
    )
    maximum = np.iinfo("int64").max
    for operation in (column.scale(2), column.offset(1)):
        with pytest.raises(OverflowError, match="range"):
            operation.apply({"id": np.array([maximum], dtype="int64")})
    exact = column.offset(0.0, dtype="int64").apply(
        {"id": np.array([maximum], dtype="int64")}
    )
    assert exact[0] == maximum
    unsigned = column.offset(1, dtype="uint64")
    with pytest.raises(OverflowError):
        unsigned.apply({"id": np.array([2**64 - 1], dtype="uint64")})
    with pytest.raises(TypeError, match="explicit dtype"):
        column.offset(-1).output_dtype("uint64")


@pytest.mark.parametrize("value", [1, 2**53 + 1])
def test_fractional_integer_transform_is_not_rounded_before_cast(value):
    column = event_source(variables="id")["id"].scale(0.5, dtype="int64")
    with pytest.raises(ValueError, match="fractional"):
        column.apply({"id": np.array([value], dtype="int64")})
    np.testing.assert_array_equal(
        column.apply({"id": np.array([2**53 + 2], dtype="int64")}), [2**52 + 1]
    )


def test_float_arithmetic_uses_inferred_or_declared_precision_and_checks_overflow():
    column = event_source(variables="t")["t"]
    chunk = {"t": np.array([3e38], dtype="float32")}
    result = column.scale(2).apply(chunk)
    assert result.dtype == np.dtype("float64")
    np.testing.assert_allclose(result, [6e38], rtol=1e-6)
    with pytest.raises(OverflowError):
        column.scale(2, dtype="float32").apply(chunk)
    with pytest.raises(OverflowError):
        column.scale(2).apply({"t": np.array([1e308])})
    with pytest.raises(ValueError, match="integral"):
        column.map(lambda values: values + 0.25, dtype="int64").apply(
            {"t": np.array([1.0])}
        )
    with pytest.raises(OverflowError, match="range"):
        column.map(lambda values: values, dtype="int64").apply(
            {"t": np.array([float(2**63)])}
        )


def test_complex_schema_is_numeric_but_sampling_and_lossy_real_casts_are_rejected(
    tmp_path,
):
    path = _write_source(
        tmp_path / "complex.nxs",
        components=[("First", _EVENTS.astype("complex128"), "p t id w L", True)],
    )
    events = event_source(variables="id")
    with SourceContext(path) as context:
        assert events.schema(context) == {"id": np.dtype("complex128")}
        with pytest.raises(TypeError, match="weights must be real"):
            events.sample(SamplingSettings(2)).schema(context)
    column = events["id"].map(lambda values: values, dtype="float64")
    with pytest.raises(TypeError, match="complex"):
        column.apply({"id": np.array([1 + 2j])})


@pytest.mark.parametrize(
    "make_operation",
    [
        lambda column: column.scale(2),
        lambda column: column.offset(1),
        lambda column: column.convert_units("s", "us"),
        lambda column: column.map_values({1: 1}, dtype="int64"),
        lambda column: column.map(lambda values: values, dtype="float64"),
    ],
)
def test_sampled_weight_operations_cannot_change_unit_weights(make_operation):
    events = event_source(variables=("p", "w", "id")).sample(
        SamplingSettings(7, weight_key="w")
    )
    with pytest.raises(ValueError, match="unit weights"):
        make_operation(events["w"])
    assert make_operation(events["p"]).source is events


def test_identity_operations_on_sampled_weights_are_safe():
    events = event_source(variables="p").sample(SamplingSettings(2))
    column = events["p"].scale(1).offset(0, dtype="int64")
    np.testing.assert_array_equal(column.apply({"p": np.ones(2)}), [1, 1])


def test_transforms_are_batch_boundary_independent():
    column = (
        event_source(variables="id")["id"]
        .map_values({0: 2, 1: 3, 2: 4}, dtype="int64")
        .scale(3)
        .offset(-1)
    )
    values = np.array([0.0, 2, 1, 2, 2, 0])
    full = column.apply({"id": values})
    batched = np.concatenate(
        [column.apply({"id": part}) for part in np.array_split(values, 4)]
    )
    np.testing.assert_array_equal(batched, full)


@pytest.mark.parametrize(
    "make_operation",
    [
        lambda column: column.scale(np.inf),
        lambda column: column.offset([1]),
        lambda column: column.scale(True),
        lambda column: column.map(None, dtype="int64"),
        lambda column: column.map(lambda value: value, dtype=None),
        lambda column: column.map_values({}, dtype=None),
        lambda column: column.map_values({}, dtype="int64", missing="zero"),
        lambda column: column.map_values({"1": 1}, dtype="int64"),
        lambda column: column.map(lambda value: value, dtype="S4"),
    ],
)
def test_invalid_transform_configuration(make_operation):
    with pytest.raises((TypeError, ValueError)):
        make_operation(event_source(variables="id")["id"])


@pytest.mark.parametrize("event_iteration", [False, True])
def test_iterators_revalidate_handles_before_next_read(source_path, event_iteration):
    context = SourceContext(source_path)
    if event_iteration:
        chunks = event_source(variables="id", read_chunk_size=1).iter_chunks(context)
    else:
        chunks = context.entry.iter_dataset("matrix", read_chunk_size=1)
    next(chunks)
    context.close()
    with pytest.raises(ValueError, match="source file is closed"):
        next(chunks)


@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("filter_zeros", [False, True])
def test_empty_and_zero_weight_streams_keep_schema_without_sampling(
    tmp_path, empty, filter_zeros
):
    rows = np.empty((0, 5)) if empty else _EVENTS.copy()
    rows[:, 3] = 0
    path = _write_source(
        tmp_path / "zero.nxs", components=[("First", rows, "p t id w L", True)]
    )
    settings = SamplingSettings(4, seed=42, weight_key="w")
    events = event_source(
        variables="id", filter_zeros=filter_zeros, weight_key="w", read_chunk_size=2
    ).sample(settings)
    with SourceContext(path) as context:
        assert events.schema(context) == dict.fromkeys(("id", "w"), np.dtype("float64"))
        chunks = list(events.iter_chunks(context))
        assert sum(len(chunk["id"]) for chunk in chunks) == (
            0 if empty or filter_zeros else 5
        )
        sampled = sample_event_chunks(chunks, settings)
    assert sampled.total_weight == 0
    assert sampled.effective_duration is None


def test_unsigned_identity_and_explicit_arithmetic_are_exact():
    column = event_source(variables="id")["id"]
    values = np.array([2**64 - 1], dtype="uint64")
    assert column.scale(1).output_dtype("uint64") == np.dtype("uint64")
    np.testing.assert_array_equal(column.scale(1).apply({"id": values}), values)
    np.testing.assert_array_equal(
        column.offset(-1, dtype="uint64").apply({"id": values}), values - 1
    )


def test_mapping_snapshots_zero_dimensional_numpy_values():
    mapped_value = np.array(5)
    column = event_source(variables="id")["id"].map_values(
        {0: mapped_value}, dtype="int64"
    )
    mapped_value[...] = 99
    np.testing.assert_array_equal(column.apply({"id": np.array([0])}), [5])


def test_complex_object_callbacks_cannot_silently_discard_imaginary_parts():
    column = event_source(variables="id")["id"].map(
        lambda values: np.array([np.complex64(1 + 2j)], dtype=object), dtype="float64"
    )
    with pytest.raises(TypeError, match="complex"):
        column.apply({"id": np.array([1])})
