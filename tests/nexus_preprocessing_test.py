# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 McCode-dev contributors (https://github.com/mccode-dev)
"""Pre-sampling acceptance tests using a pixel-free McStas absorption logger."""

import importlib
from dataclasses import FrozenInstanceError, asdict

import h5py
import numpy as np
import pytest

import mcstastox
from mcstastox.ReadNeXus import McStasNeXus

_VARIABLES = ("p", "x", "y", "z", "t")
_RAW = np.array(
    [
        [0, -0.75, 0.2, 1.0, 0.001],
        [2, -0.25, 0.2, 1.0, 0.002],
        [3, 0.25, 0.2, 1.0, 0.003],
        [5, 0.75, 0.2, 1.0, 0.004],
        [7, -0.25, 1.2, 1.0, 0.005],
        [11, 0.25, -0.2, 1.5, 0.006],
        [13, -0.25, -0.2, 1.5, 0.007],
        [17, 0.25, -0.2, 3.0, 0.008],
        [0, 0.25, 0.2, 1.5, 0.009],
    ],
    dtype="float64",
)
_SCHEMA = {"id": "int32", "t": "float64", "p": "float64", "depth": "float32"}
_STATISTICS = (
    "total_weight",
    "total_weight_variance",
    "effective_duration",
    "effective_duration_variance",
)


def _write_logger(path, rows=_RAW, *, variables=_VARIABLES):
    """The logger deliberately has neither an id column nor a BINS group."""
    with h5py.File(path, "w") as handle:
        entry = handle.create_group("entry1")
        entry.create_group("data")
        simulation = entry.create_group("simulation")
        simulation.attrs["program"] = np.bytes_("3.8.6")
        simulation.create_group("Param")
        output = entry.create_group("instrument/components/0000_Absorption/output")
        info = output.create_group("event_data")
        info.attrs["variables"] = np.bytes_(" ".join(variables) + " ")
        info.create_dataset("events", data=rows)
    return path


@pytest.fixture
def logger_path(tmp_path):
    return _write_logger(tmp_path / "absorption.nxs")


def _mask(chunk):
    return (
        (np.abs(chunk["x"]) < 0.5)
        & (np.abs(chunk["y"]) < 0.5)
        & (chunk["z"] >= 1)
        & (chunk["z"] < 2)
    )


def _voxelize(chunk):
    keep = _mask(chunk)
    return {
        "id": 100 + (chunk["x"][keep] > 0) + 2 * (chunk["y"][keep] < 0),
        "t": chunk["t"][keep] + chunk["z"][keep] * 0.0001,
        "p": chunk["p"][keep],
        "depth": chunk["z"][keep],
    }


def _oracle_chunks(rows=_RAW, chunk_size=2):
    # Known geometry results, independent of the callback's acceptance calculation.
    accepted = np.array([False, True, True, False, False, True, True, False, True])
    pixels = np.array([0, 100, 101, 0, 0, 103, 102, 0, 101], dtype="int32")
    for start in range(0, len(rows), chunk_size):
        stop = start + chunk_size
        keep = accepted[start:stop]
        block = rows[start:stop][keep]
        yield {
            "id": pixels[start:stop][keep],
            "t": block[:, 4] + block[:, 3] * 0.0001,
            "p": block[:, 0].copy(),
            "depth": block[:, 3].astype("float32"),
        }


def _prepared(callback=_voxelize, **kwargs):
    return mcstastox.event_source(
        component_name="Absorption", variables=_VARIABLES, read_chunk_size=2, **kwargs
    ).preprocess(callback, schema=_SCHEMA, description="voxel acceptance")


def _spy_sampler(monkeypatch):
    writer = importlib.import_module("mcstastox.NexusWrite")
    original = writer.sample_event_chunks
    calls = []

    def observed(chunks, settings):
        result = original(chunks, settings)
        calls.append((settings, result))
        return result

    monkeypatch.setattr(writer, "sample_event_chunks", observed)
    return calls


def _forbidden(*args, **kwargs):
    pytest.fail("Metadata inspection must not execute callbacks or read payloads")


def test_descriptors_and_raw_processed_schemas_are_immutable_and_metadata_only(
    logger_path, monkeypatch
):
    declared = dict(_SCHEMA)
    raw = mcstastox.event_source(
        component_name="Absorption", variables=("x", "y", "z", "t"), read_chunk_size=2
    )
    prepared = raw.preprocess(_forbidden, schema=declared, description="voxelize")
    sampled = prepared.sample(mcstastox.SamplingSettings(7, seed=42))
    declared["id"] = "float32"
    declared["extra"] = "int64"

    assert raw.preprocessors == ()
    assert isinstance(prepared.preprocessors, tuple)
    assert len(prepared.preprocessors) == 1
    stage = prepared.preprocessors[0]
    assert isinstance(stage, mcstastox.ChunkOperation)
    assert sampled.preprocessors == prepared.preprocessors
    assert sampled.preprocessors[0] is stage
    assert stage.callback is _forbidden
    assert stage.inputs is None
    assert stage.schema == {key: np.dtype(value) for key, value in _SCHEMA.items()}
    assert (
        raw.variables == prepared.variables == sampled.variables == ("x", "y", "z", "t")
    )
    assert raw.output_variables == raw.variables
    assert sampled.output_variables == tuple(_SCHEMA)
    assert sampled.required_variables == ("x", "y", "z", "t", "p")
    assert sampled["id"].execution_source is sampled
    with pytest.raises(KeyError, match="not declared"):
        sampled["x"]
    with pytest.raises(TypeError):
        stage.schema["id"] = np.dtype("float64")
    with pytest.raises(FrozenInstanceError):
        stage.callback = _voxelize
    with pytest.raises(FrozenInstanceError):
        prepared.preprocessors = ()
    with pytest.raises((ValueError, TypeError), match="before sample"):
        sampled.preprocess(_forbidden, schema=_SCHEMA)

    template = mcstastox.create_empty_template()
    template.add_dataset("/ids").copy_from(sampled["id"])
    template.add_dataset("/table").copy_from(sampled, columns=("id", "t", "depth"))
    template.add_dataset("/duration").set_from_result(sampled, _forbidden)
    with mcstastox.SourceContext(logger_path) as context:
        monkeypatch.setattr(h5py.Dataset, "__getitem__", _forbidden)
        monkeypatch.setattr(h5py.Dataset, "__array__", _forbidden)
        monkeypatch.setattr(h5py.Dataset, "read_direct", _forbidden)
        monkeypatch.setattr(context.mcstas, "iter_event_data", _forbidden)
        assert context.mcstas.get_components_with_ids() == []
        assert raw.raw_schema(context) == dict.fromkeys(
            raw.variables, np.dtype("float64")
        )
        assert raw.schema(context) == raw.raw_schema(context)
        assert sampled.raw_schema(context) == dict.fromkeys(
            sampled.required_variables, np.dtype("float64")
        )
        assert sampled.schema(context) == dict(stage.schema)
        inspected = sampled.schema(context)
        inspected["id"] = np.dtype("int8")
        assert sampled.schema(context)["id"] == np.dtype("int32")
        assert raw.row_count(context) == len(_RAW)
        assert prepared.row_count(context) is sampled.row_count(context) is None
        template.validate(source=context).raise_if_errors()
        assert "voxelize" in template.summary()
        assert "declared output schema" in repr(prepared)
        assert "voxelize" in repr(stage)
        with pytest.raises((ValueError, TypeError), match="No event components"):
            mcstastox.event_source(variables=("x", "t")).preprocess(
                _forbidden, schema=_SCHEMA
            ).schema(context)
        with pytest.raises((ValueError, TypeError), match="exact event variable 'id'"):
            mcstastox.event_source(component_name="Absorption").preprocess(
                _forbidden, schema=_SCHEMA
            ).raw_schema(context)


def test_callbacks_receive_complete_private_read_only_chunks(logger_path, monkeypatch):
    original = McStasNeXus.iter_event_data
    reader_chunks = []
    snapshots = []
    calls = []

    def observed(reader, *args, **kwargs):
        for chunk in original(reader, *args, **kwargs):
            reader_chunks.append(chunk)
            snapshots.append({key: value.copy() for key, value in chunk.items()})
            yield chunk

    def prepare(chunk):
        assert isinstance(chunk, dict)
        assert set(chunk) == set(_VARIABLES)
        calls.append(len(chunk["t"]))
        for values in chunk.values():
            assert not values.flags.writeable
            assert all(
                not np.shares_memory(values, raw) for raw in reader_chunks[-1].values()
            )
        result = _voxelize(chunk)
        with pytest.raises(ValueError, match="read-only"):
            chunk["x"][0] = 99
        # Even deliberately re-enabling writes cannot corrupt the reader's block.
        chunk["x"].flags.writeable = True
        chunk["x"][:] = 99
        chunk.clear()
        return result

    before = logger_path.read_bytes()
    monkeypatch.setattr(McStasNeXus, "iter_event_data", observed)
    events = _prepared(prepare)
    with mcstastox.SourceContext(logger_path) as context:
        chunks = list(events.iter_chunks(context))
    assert calls == [2, 2, 2, 2, 1]
    assert len(reader_chunks) == len(chunks) == 5
    for original_chunk, snapshot in zip(reader_chunks, snapshots, strict=True):
        for key in _VARIABLES:
            np.testing.assert_array_equal(original_chunk[key], snapshot[key])
    oracle = list(_oracle_chunks())
    for key, dtype in _SCHEMA.items():
        assert all(chunk[key].dtype == np.dtype(dtype) for chunk in chunks)
        np.testing.assert_array_equal(
            np.concatenate([chunk[key] for chunk in chunks]),
            np.concatenate([chunk[key] for chunk in oracle]),
        )
    assert logger_path.read_bytes() == before


@pytest.mark.parametrize("ordered", [False, True])
def test_preprocessing_precedes_one_shared_sample_columns_table_and_results(
    tmp_path, logger_path, monkeypatch, ordered
):
    settings = mcstastox.SamplingSettings(23, seed=17, ordered=ordered)
    expected = mcstastox.sample_event_chunks(_oracle_chunks(), settings)
    calls = _spy_sampler(monkeypatch)
    preprocessing_calls = []

    def prepare(chunk):
        preprocessing_calls.append(len(chunk["t"]))
        assert set(chunk) == set(_VARIABLES)
        return _voxelize(chunk)

    events = (
        mcstastox.event_source(
            component_name="Absorption",
            variables=("x", "y", "z", "t"),
            read_chunk_size=2,
        )
        .preprocess(prepare, schema=_SCHEMA, description="voxel acceptance")
        .sample(settings)
    )
    projected = events.transform(
        lambda chunk: {"pixel": chunk["id"] + 1000, "time_us": chunk["t"] * 1e6},
        inputs=("id", "t"),
        schema={"pixel": "int64", "time_us": "float64"},
        description="post-sampling projection",
    )
    template = mcstastox.create_empty_template()
    template.add_dataset("/ids").copy_from(events["id"])
    template.add_dataset("/time_us").copy_from(events["t"].convert_units("s", "us"))
    template.add_dataset("/weights").copy_from(events["p"])
    template.add_dataset("/table").copy_from(events, columns=("t", "p", "id", "depth"))
    template.add_dataset("/pixels").copy_from(projected["pixel"])
    for name in _STATISTICS:
        template.add_dataset("/" + name).set_from_result(events, name)
    template.attrs["selected"].set_from_result(projected, "n_selected")
    before = logger_path.read_bytes()
    output = tmp_path / "sampled.nxs"

    report = mcstastox.write_nexus(
        template, output, source=logger_path, read_chunk_size=100, write_chunk_size=3
    )

    assert preprocessing_calls == [2, 2, 2, 2, 1]
    assert len(calls) == 1
    assert calls[0][0] is settings
    for name in _SCHEMA:
        np.testing.assert_array_equal(calls[0][1][name], expected[name])
    assert set(report.sources) == set(report.preprocessing) == {events}
    assert report.preprocessing[events] is events.preprocessors
    assert report.preprocessing[events][0].callback is prepare
    assert report.preprocessing[events][0].schema["id"] == np.dtype("int32")
    result = report.sources[events]
    assert result.n_selected == result.n_written == 23
    assert result.total_weight == 29
    assert result.total_weight_variance == 303
    assert result.effective_duration == pytest.approx(23 / 29)
    assert result.effective_duration_variance == pytest.approx(
        (23 / 29) ** 2 * 303 / 29**2
    )
    assert set(asdict(result)) == {"n_written", "n_selected", *_STATISTICS}
    assert all(value is None or np.isscalar(value) for value in asdict(result).values())
    for name in _STATISTICS:
        assert getattr(result, name) == getattr(expected, name)
    with h5py.File(output, "r") as handle:
        np.testing.assert_array_equal(handle["ids"][:], expected["id"])
        np.testing.assert_array_equal(handle["time_us"][:], expected["t"] * 1e6)
        np.testing.assert_array_equal(handle["weights"][:], np.ones(23))
        np.testing.assert_array_equal(handle["pixels"][:], expected["id"] + 1000)
        np.testing.assert_array_equal(
            handle["table"][:],
            np.column_stack([expected[key] for key in ("t", "p", "id", "depth")]),
        )
        assert handle["ids"].dtype == np.dtype("int32")
        assert handle["weights"].dtype == np.dtype("float64")
        assert handle.attrs["selected"] == 23
        for name in _STATISTICS:
            assert handle[name].shape == ()
            assert handle[name][()] == getattr(expected, name)
    for path in ("/ids", "/time_us", "/weights", "/table", "/pixels"):
        destination = report.destinations[path]
        assert destination.source is events
        assert destination.n_written == 23
        assert destination.shape == ((23, 4) if path == "/table" else (23,))
    assert report.destinations["/ids"].chunk_operations == ()
    assert report.destinations["/pixels"].chunk_operations == (projected.operation,)
    assert (
        "post-sampling projection" in report.destinations["/pixels"].transformations[0]
    )
    assert logger_path.read_bytes() == before
    assert set(tmp_path.iterdir()) == {logger_path, output}


@pytest.mark.parametrize("seed", [37, None], ids=["seeded", "unseeded"])
def test_reused_template_executes_fresh_and_is_output_batch_invariant(
    tmp_path, logger_path, monkeypatch, seed
):
    changed_rows = _RAW.copy()
    changed_rows[:, 0] *= 2
    changed_rows[:, 4] += 0.05
    changed_path = _write_logger(tmp_path / "changed.nxs", changed_rows)
    before = {path: path.read_bytes() for path in (logger_path, changed_path)}
    calls = _spy_sampler(monkeypatch)
    preprocessing_calls = []

    def prepare(chunk):
        preprocessing_calls.append(len(chunk["t"]))
        return _voxelize(chunk)

    events = _prepared(prepare).sample(mcstastox.SamplingSettings(19, seed=seed))
    template = mcstastox.create_empty_template()
    template.add_dataset("/ids").copy_from(events["id"])
    template.add_dataset("/times").copy_from(events["t"].convert_units("s", "us"))
    template.add_dataset("/table").copy_from(events, columns=("id", "p", "depth"))
    summary = template.summary()
    payloads = []
    reports = []
    for index, (source, batch_size) in enumerate(
        ((logger_path, 1), (logger_path, 7), (changed_path, 3))
    ):
        output = tmp_path / f"reuse-{index}.nxs"
        report = mcstastox.write_nexus(
            template, output, source=source, write_chunk_size=batch_size
        )
        reports.append(report)
        assert len(calls) == index + 1
        assert preprocessing_calls == [2, 2, 2, 2, 1] * (index + 1)
        assert template.summary() == summary
        sampled = calls[index][1]
        with h5py.File(output, "r") as handle:
            values = {key: handle[key][:] for key in ("ids", "times", "table")}
        payloads.append(values)
        np.testing.assert_array_equal(values["ids"], sampled["id"])
        np.testing.assert_array_equal(values["times"], sampled["t"] * 1e6)
        np.testing.assert_array_equal(
            values["table"],
            np.column_stack([sampled[key] for key in ("id", "p", "depth")]),
        )
        assert report.preprocessing[events] == events.preprocessors
    assert len({id(sampled) for _, sampled in calls}) == 3
    assert reports[0].sources[events] is not reports[1].sources[events]
    if seed is not None:
        for key in payloads[0]:
            np.testing.assert_array_equal(payloads[0][key], payloads[1][key])
        assert reports[0].sources[events] == reports[1].sources[events]
    assert np.all(payloads[2]["times"] > 50_000)
    assert reports[2].sources[events].total_weight == 58
    for path, contents in before.items():
        assert path.read_bytes() == contents


def test_independent_samples_of_same_preprocessed_source_are_not_coalesced(
    tmp_path, logger_path, monkeypatch
):
    calls = _spy_sampler(monkeypatch)
    preprocessing_calls = []

    def prepare(chunk):
        preprocessing_calls.append(len(chunk["t"]))
        return _voxelize(chunk)

    prepared = _prepared(prepare)
    settings = mcstastox.SamplingSettings(13, seed=42)
    first, second = prepared.sample(settings), prepared.sample(settings)
    assert len({prepared, first, second}) == 3
    template = mcstastox.create_empty_template()
    template.add_dataset("/first").copy_from(first["id"])
    template.add_dataset("/second").copy_from(second["id"])
    output = tmp_path / "independent.nxs"
    report = mcstastox.write_nexus(template, output, source=logger_path)
    assert len(calls) == 2
    assert all(call_settings is settings for call_settings, _ in calls)
    assert calls[0][1] is not calls[1][1]
    assert preprocessing_calls == [2, 2, 2, 2, 1] * 2
    assert set(report.sources) == set(report.preprocessing) == {first, second}
    with h5py.File(output, "r") as handle:
        np.testing.assert_array_equal(handle["first"][:], calls[0][1]["id"])
        np.testing.assert_array_equal(handle["second"][:], calls[1][1]["id"])


@pytest.mark.parametrize(("read_size", "write_size"), [(1, 5), (4, 1)])
def test_unsampled_chained_acceptance_streams_once_and_is_partition_independent(
    tmp_path, logger_path, monkeypatch, read_size, write_size
):
    first_calls, second_calls = [], []
    updates = []
    retained_rows = 0

    def first(chunk):
        written = max(
            (update.rows for update in updates if update.phase == "write"), default=0
        )
        assert written == retained_rows, "Unsampled writing collected input chunks"
        first_calls.append(len(chunk["t"]))
        return _voxelize(chunk)

    def second(chunk):
        nonlocal retained_rows
        second_calls.append(len(chunk["id"]))
        assert set(chunk) == set(_SCHEMA)
        assert all(not value.flags.writeable for value in chunk.values())
        assert chunk["id"].dtype == np.dtype("int32")
        assert chunk["depth"].dtype == np.dtype("float32")
        keep = chunk["id"] >= 102
        retained_rows += np.count_nonzero(keep)
        return {"pixel": chunk["id"][keep] + 1000, "arrival": chunk["t"][keep]}

    events = (
        mcstastox.event_source(
            component_name="Absorption", variables=_VARIABLES, read_chunk_size=read_size
        )
        .preprocess(first, schema=_SCHEMA, description="geometry")
        .preprocess(
            second,
            schema={"pixel": "uint16", "arrival": "float32"},
            description="pixels",
        )
    )
    assert events.variables == _VARIABLES
    assert events.output_variables == ("pixel", "arrival")
    template = mcstastox.create_empty_template()
    template.add_dataset("/pixels").copy_from(events["pixel"])
    template.add_dataset("/arrival").copy_from(events["arrival"])
    template.add_dataset("/table").copy_from(events, columns=("arrival", "pixel"))
    template.add_dataset("/count").set_from_result(events, "n_written")
    writer = importlib.import_module("mcstastox.NexusWrite")
    monkeypatch.setattr(writer, "sample_event_chunks", _forbidden)
    output = tmp_path / "unsampled.nxs"
    report = mcstastox.write_nexus(
        template,
        output,
        source=logger_path,
        write_chunk_size=write_size,
        progress=updates.append,
    )
    assert first_calls == [
        min(read_size, len(_RAW) - i) for i in range(0, 9, read_size)
    ]
    assert len(second_calls) == len(first_calls)
    assert sum(second_calls) == 5
    with h5py.File(output, "r") as handle:
        np.testing.assert_array_equal(handle["pixels"][:], [1103, 1102])
        times = np.array([0.00615, 0.00715], dtype="float32")
        np.testing.assert_array_equal(handle["arrival"][:], times)
        np.testing.assert_array_equal(
            handle["table"][:], np.column_stack((times, [1103, 1102]))
        )
        assert handle["pixels"].dtype == np.dtype("uint16")
        assert handle["arrival"].dtype == np.dtype("float32")
        assert handle["count"][()] == 2
    assert report.sources[events].n_written == 2
    assert report.sources[events].n_selected is None
    assert all(getattr(report.sources[events], name) is None for name in _STATISTICS)
    assert report.preprocessing[events] == events.preprocessors
    assert tuple(stage.callback for stage in report.preprocessing[events]) == (
        first,
        second,
    )


def test_zero_filter_precedes_callbacks_and_hidden_custom_weight_survives_each_stage(
    tmp_path, monkeypatch
):
    weights = np.array([np.nan, 0, 4, -2, np.inf, 6, 7, 8, np.nan])
    rows = np.column_stack((_RAW, weights))
    source = _write_logger(tmp_path / "custom.nxs", rows, variables=(*_VARIABLES, "w"))
    first_inputs, second_inputs = [], []

    def first(chunk):
        assert set(chunk) == {"x", "y", "z", "t", "p", "w"}
        assert np.all(chunk["p"] != 0)
        first_inputs.append(chunk["w"].copy())
        prepared = _voxelize(chunk)
        return {"id": prepared["id"], "t": prepared["t"], "w": chunk["w"][_mask(chunk)]}

    def second(chunk):
        assert set(chunk) == {"id", "t", "w"}
        second_inputs.append(chunk["w"].copy())
        return {"pixel": chunk["id"], "time": chunk["t"], "w": chunk["w"]}

    events = (
        mcstastox.event_source(
            component_name="Absorption",
            variables=("x", "y", "z", "t"),
            read_chunk_size=3,
            filter_zeros=True,
            weight_key="p",
        )
        .preprocess(first, schema={"id": "int32", "t": "float64", "w": "float64"})
        .preprocess(
            second, schema={"pixel": "int32", "time": "float64", "w": "float64"}
        )
        .sample(mcstastox.SamplingSettings(17, seed=9, weight_key="w"))
    )
    assert events.required_variables == ("x", "y", "z", "t", "w", "p")
    expected_chunks = []
    for chunk, retained in zip(
        _oracle_chunks(chunk_size=3), ([1, 2], [5], [6, 8]), strict=True
    ):
        keep = _RAW[retained, 0] != 0
        expected_chunks.append(
            {
                "pixel": chunk["id"][keep],
                "time": chunk["t"][keep],
                "w": weights[retained][keep],
            }
        )
    expected = mcstastox.sample_event_chunks(expected_chunks, events.sampling)
    calls = _spy_sampler(monkeypatch)
    template = mcstastox.create_empty_template()
    template.add_dataset("/pixels").copy_from(events["pixel"])
    template.add_dataset("/table").copy_from(events, columns=("pixel", "time", "w"))
    output = tmp_path / "custom-output.nxs"
    before = source.read_bytes()
    report = mcstastox.write_nexus(template, output, source=source, write_chunk_size=2)
    assert len(calls) == 1
    assert [values.tolist() for values in first_inputs] == [
        [0, 4],
        [-2, np.inf, 6],
        [7, 8],
    ]
    assert [values.tolist() for values in second_inputs] == [[0, 4], [6], [7]]
    assert report.sources[events].total_weight == 17
    assert report.sources[events].total_weight_variance == 101
    for name in _STATISTICS:
        assert getattr(report.sources[events], name) == getattr(expected, name)
    with h5py.File(output, "r") as handle:
        np.testing.assert_array_equal(handle["pixels"][:], expected["pixel"])
        np.testing.assert_array_equal(
            handle["table"][:],
            np.column_stack([expected[key] for key in ("pixel", "time", "w")]),
        )
        np.testing.assert_array_equal(handle["table"][:, 2], np.ones(17))
    assert source.read_bytes() == before


@pytest.mark.parametrize("sampled", [False, True], ids=["unsampled", "sampled"])
@pytest.mark.parametrize(
    "kind", ["empty", "rejected", "zero-accepted", "zero-filtered"]
)
def test_empty_rejected_and_zero_streams_have_declared_typed_outputs(
    tmp_path, monkeypatch, sampled, kind
):
    rows = _RAW[:0].copy() if kind == "empty" else _RAW.copy()
    if kind == "rejected":
        rows[:, 1] = 10
    if kind.startswith("zero-"):
        rows[:, 0] = 0
    source = _write_logger(tmp_path / "empty-or-zero.nxs", rows)
    calls = []

    def prepare(chunk):
        calls.append(len(chunk["t"]))
        return _voxelize(chunk)

    events = _prepared(prepare, filter_zeros=kind == "zero-filtered")
    if sampled:
        events = events.sample(mcstastox.SamplingSettings(11, seed=42))
    else:
        writer = importlib.import_module("mcstastox.NexusWrite")
        monkeypatch.setattr(writer, "sample_event_chunks", _forbidden)
    template = mcstastox.create_empty_template()
    for name in _SCHEMA:
        template.add_dataset("/" + name).copy_from(events[name])
    template.add_dataset("/table").copy_from(events, columns=("id", "t", "p", "depth"))
    template.add_dataset("/duration").set_from_result(
        events,
        lambda result: (
            0.0 if result.effective_duration is None else result.effective_duration
        ),
    )
    output = tmp_path / "typed.nxs"
    report = mcstastox.write_nexus(template, output, source=source, write_chunk_size=1)
    count = 5 if kind == "zero-accepted" and not sampled else 0
    assert calls == ([] if kind in {"empty", "zero-filtered"} else [2, 2, 2, 2, 1])
    with h5py.File(output, "r") as handle:
        for name, dtype in _SCHEMA.items():
            assert handle[name].shape == (count,)
            assert handle[name].dtype == np.dtype(dtype)
            assert report.destinations["/" + name].n_written == count
        assert handle["table"].shape == (count, 4)
        assert handle["table"].dtype == np.dtype("float64")
        assert handle["duration"][()] == 0
        if count:
            np.testing.assert_array_equal(handle["p"][:], np.zeros(5))
            np.testing.assert_array_equal(handle["id"][:], [100, 101, 103, 102, 101])
    result = report.sources[events]
    assert result.n_written == count
    assert result.n_selected == (0 if sampled else None)
    assert result.total_weight == (0 if sampled else None)
    assert result.total_weight_variance == (0 if sampled else None)
    assert result.effective_duration is result.effective_duration_variance is None
    assert report.preprocessing[events] is events.preprocessors
    assert set(tmp_path.iterdir()) == {source, output}


@pytest.mark.parametrize(
    ("stage_index", "dtype"),
    [(0, None), (1, None), (0, "complex128"), (1, "complex128")],
)
def test_sampling_requires_real_custom_weight_in_every_declared_stage(
    stage_index, dtype
):
    schemas = [{"id": "int32", "w": "float64"}, {"pixel": "int32", "w": "float64"}]
    if dtype is None:
        del schemas[stage_index]["w"]
    else:
        schemas[stage_index]["w"] = dtype
    events = mcstastox.event_source(variables="x")
    for schema in schemas:
        events = events.preprocess(_forbidden, schema=schema)
    with pytest.raises((ValueError, TypeError), match=r"(?i)(weight|retain|real)"):
        events.sample(mcstastox.SamplingSettings(3, weight_key="w"))
    assert events.sampling is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"callback": None, "schema": _SCHEMA},
        {"callback": _forbidden, "schema": []},
        {"callback": _forbidden, "schema": {}},
        {"callback": _forbidden, "schema": {"bad name": "float64"}},
        {"callback": _forbidden, "schema": {"id": "bool"}},
        {"callback": _forbidden, "schema": {"id": "S4"}},
        {"callback": _forbidden, "schema": _SCHEMA, "description": 7},
    ],
)
def test_invalid_preprocessing_declarations_fail_without_callback_execution(kwargs):
    with pytest.raises((ValueError, TypeError)):
        mcstastox.event_source().preprocess(**kwargs)


@pytest.mark.parametrize(
    "case",
    [
        "not-a-mapping",
        "missing-key",
        "extra-key",
        "unequal-lengths",
        "scalar",
        "matrix",
        "excess-rows",
        "text",
        "fractional-id",
        "nonfinite-id",
        "integer-overflow",
        "float-overflow",
        "complex-to-real",
    ],
)
def test_malformed_callback_output_fails_atomically_with_stage_context(tmp_path, case):
    source = _write_logger(tmp_path / "bad-output.nxs", _RAW[1:3])
    before = source.read_bytes()
    calls = []

    def bad(chunk):
        calls.append(len(chunk["t"]))
        result = {"id": np.array([1, 2]), "t": chunk["t"], "p": chunk["p"]}
        if case == "not-a-mapping":
            return None
        if case == "missing-key":
            del result["p"]
        elif case == "extra-key":
            result["extra"] = chunk["x"]
        elif case == "unequal-lengths":
            result["id"] = np.array([1])
        elif case == "scalar":
            result["id"] = np.array(1)
        elif case == "matrix":
            result["id"] = np.array([[1], [2]])
        elif case == "excess-rows":
            result = {key: np.append(value, value[-1]) for key, value in result.items()}
        elif case == "text":
            result["id"] = np.array(["one", "two"])
        elif case == "fractional-id":
            result["id"] = np.array([1.5, 2])
        elif case == "nonfinite-id":
            result["id"] = np.array([np.nan, 2])
        elif case == "integer-overflow":
            result["id"] = np.array([2**15, 2])
        elif case == "float-overflow":
            result["t"] = np.array([1e300, 0])
        elif case == "complex-to-real":
            result["t"] = np.array([1 + 2j, 0])
        return result

    events = mcstastox.event_source(
        component_name="Absorption", variables=_VARIABLES, read_chunk_size=2
    ).preprocess(
        bad,
        schema={"id": "int16", "t": "float32", "p": "float64"},
        description="bad logger output",
    )
    template = mcstastox.create_empty_template()
    template.add_dataset("/ids").copy_from(events["id"])
    output = tmp_path / "must-not-exist.nxs"
    with pytest.raises(
        (ValueError, TypeError, OverflowError), match="Preprocessing stage 1"
    ) as error:
        mcstastox.write_nexus(template, output, source=source)
    assert "bad logger output" in str(error.value)
    assert error.value.__cause__ is not None
    assert calls == [2]
    assert not output.exists()
    assert set(tmp_path.iterdir()) == {source}
    assert source.read_bytes() == before


@pytest.mark.parametrize(
    "invalid", [-1.0, np.nan, np.inf], ids=["negative", "nan", "infinite"]
)
def test_sampler_rejects_invalid_retained_custom_weights_atomically(tmp_path, invalid):
    rows = np.column_stack((_RAW, np.ones(len(_RAW))))
    source = _write_logger(tmp_path / "weights.nxs", rows, variables=(*_VARIABLES, "w"))
    before = source.read_bytes()

    def prepare(chunk):
        assert "w" in chunk
        result = _voxelize(chunk)
        return {"id": result["id"], "w": np.full(len(result["id"]), invalid)}

    events = (
        mcstastox.event_source(
            component_name="Absorption", variables=_VARIABLES, read_chunk_size=2
        )
        .preprocess(prepare, schema={"id": "int32", "w": "float64"})
        .sample(mcstastox.SamplingSettings(7, seed=42, weight_key="w"))
    )
    template = mcstastox.create_empty_template()
    template.add_dataset("/ids").copy_from(events["id"])
    output = tmp_path / "invalid-weight.nxs"
    with pytest.raises((ValueError, TypeError), match=r"(?i)(weight|finite|negative)"):
        mcstastox.write_nexus(template, output, source=source)
    assert not output.exists()
    assert set(tmp_path.iterdir()) == {source}
    assert source.read_bytes() == before


@pytest.mark.parametrize("dtype", ["int32", "float32"])
def test_sampled_weight_output_dtype_is_float64_not_preprocessing_dtype(
    tmp_path, logger_path, dtype
):
    schema = {**_SCHEMA, "p": dtype}
    events = (
        mcstastox.event_source(
            component_name="Absorption", variables=_VARIABLES, read_chunk_size=2
        )
        .preprocess(_voxelize, schema=schema)
        .sample(mcstastox.SamplingSettings(7, seed=42))
    )
    template = mcstastox.create_empty_template()
    template.add_dataset("/weights").copy_from(events["p"])
    output = tmp_path / "unit-weights.nxs"
    report = mcstastox.write_nexus(template, output, source=logger_path)
    with h5py.File(output, "r") as handle:
        assert handle["weights"].dtype == np.dtype("float64")
        np.testing.assert_array_equal(handle["weights"][:], np.ones(7))
    assert report.destinations["/weights"].dtype == np.dtype("float64")


@pytest.mark.parametrize("existing", [False, True], ids=["new-output", "overwrite"])
@pytest.mark.parametrize("failing_stage", [1, 2])
def test_callback_failure_after_partial_write_preserves_inputs_and_existing_output(
    tmp_path, logger_path, existing, failing_stage
):
    output = tmp_path / "atomic.nxs"
    if existing:
        with h5py.File(output, "w") as handle:
            handle.create_dataset("previous", data=[123, 456])
    previous = output.read_bytes() if existing else None
    before = logger_path.read_bytes()
    calls = [0, 0]
    updates = []

    def first(chunk):
        calls[0] += 1
        if failing_stage == 1 and calls[0] == 2:
            raise RuntimeError("deliberate callback failure")
        return _voxelize(chunk)

    def second(chunk):
        calls[1] += 1
        if failing_stage == 2 and calls[1] == 2:
            raise RuntimeError("deliberate callback failure")
        return chunk

    events = _prepared(first).preprocess(
        second, schema=_SCHEMA, description="second stage"
    )
    template = mcstastox.create_empty_template()
    template.add_dataset("/ids").copy_from(events["id"])
    template.add_dataset("/table").copy_from(events, columns=("id", "t", "p"))
    with pytest.raises(
        RuntimeError, match=f"Preprocessing stage {failing_stage}"
    ) as error:
        mcstastox.write_nexus(
            template,
            output,
            source=logger_path,
            overwrite=existing,
            write_chunk_size=1,
            progress=updates.append,
        )
    assert "deliberate callback failure" in str(error.value)
    assert isinstance(error.value.__cause__, RuntimeError)
    assert any(update.phase == "write" and update.rows > 0 for update in updates)
    assert calls == ([2, 1] if failing_stage == 1 else [2, 2])
    assert logger_path.read_bytes() == before
    if existing:
        assert output.read_bytes() == previous
    else:
        assert not output.exists()
    assert set(tmp_path.iterdir()) == (
        {logger_path, output} if existing else {logger_path}
    )


@pytest.mark.parametrize(
    "failure", ["missing-raw-column", "undeclared-table-column", "processed-schema"]
)
def test_writer_preflight_uses_processed_schema_without_payload_or_callbacks(
    tmp_path, logger_path, monkeypatch, failure
):
    variables = (
        (*_VARIABLES, "absent") if failure == "missing-raw-column" else _VARIABLES
    )
    events = (
        mcstastox.event_source(component_name="Absorption", variables=variables)
        .preprocess(_forbidden, schema=_SCHEMA)
        .sample(mcstastox.SamplingSettings(7))
    )
    template = mcstastox.create_empty_template()
    columns = ("id", "x") if failure == "undeclared-table-column" else ("id", "t")
    template.add_dataset("/table").copy_from(events, columns=columns)
    output = tmp_path / "preflight.nxs"
    before = logger_path.read_bytes()

    def stop_before_output(*args, **kwargs):
        raise RuntimeError("preflight complete")

    with mcstastox.SourceContext(logger_path) as context:
        with monkeypatch.context() as metadata_only:
            metadata_only.setattr(h5py.Dataset, "__getitem__", _forbidden)
            metadata_only.setattr(h5py.Dataset, "__array__", _forbidden)
            metadata_only.setattr(h5py.Dataset, "read_direct", _forbidden)
            metadata_only.setattr(context.mcstas, "iter_event_data", _forbidden)
            writer = importlib.import_module("mcstastox.NexusWrite")
            metadata_only.setattr(writer.tempfile, "mkstemp", stop_before_output)
            exception = (
                RuntimeError
                if failure == "processed-schema"
                else (ValueError, TypeError)
            )
            message = (
                "preflight complete"
                if failure == "processed-schema"
                else "Destination /table"
            )
            with pytest.raises(exception, match=message):
                mcstastox.write_nexus(template, output, source=context)
        assert context.raw.id.valid
    assert logger_path.read_bytes() == before
    assert not output.exists()
    assert set(tmp_path.iterdir()) == {logger_path}
