# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 McCode-dev contributors (https://github.com/mccode-dev)
"""Synthetic coverage of post-sampling, row-preserving dictionary projections."""

import importlib
from dataclasses import FrozenInstanceError

import h5py
import numpy as np
import pytest

import mcstastox
from mcstastox import (
    ChunkOperation,
    EventColumn,
    EventSource,
    EventTransform,
    SamplingSettings,
    SourceContext,
    create_empty_template,
    event_source,
    write_nexus,
)
from mcstastox.ReadNeXus import McStasNeXus

_VARIABLES = ("p", "t", "id", "x", "y", "L", "w")


def _event_rows(ids):
    ids = np.asarray(ids)
    return np.column_stack(
        (
            ids % 4,
            (ids + 1) / 8,
            ids,
            ids - 3,
            ids % 3 - 1,
            ids + 2,
            ids % 5,
        )
    ).astype("float64")


@pytest.fixture
def source_file(tmp_path):
    path = tmp_path / "source.nxs"
    zero = _event_rows(range(4))
    zero[:, (0, 6)] = 0
    components = {
        "First": _event_rows(range(9)),
        "Second": _event_rows(range(20, 26)),
        "Empty": np.empty((0, len(_VARIABLES))),
        "Zero": zero,
    }
    with h5py.File(path, "w") as handle:
        entry = handle.create_group("entry1")
        entry.create_group("data")
        simulation = entry.create_group("simulation")
        simulation.attrs["program"] = np.bytes_("3.8.6")
        simulation.create_group("Param")
        entry.create_dataset("title", data=np.bytes_("Synthetic chunk transforms"))
        groups = entry.create_group("instrument/components")
        for index, (name, rows) in enumerate(components.items()):
            output = groups.create_group(f"{index:04d}_{name}/output")
            output.create_group("BINS")
            info = output.create_group("event_data")
            info.attrs["variables"] = np.bytes_(" ".join(_VARIABLES))
            info.create_dataset("events", data=rows)
    return path


@pytest.fixture
def sampler_calls(monkeypatch):
    sampling = importlib.import_module("mcstastox.Sampling")
    writer = importlib.import_module("mcstastox.NexusWrite")
    original = sampling.sample_event_chunks
    calls = []

    def observed(chunks, settings):
        result = original(chunks, settings)
        calls.append((settings, result))
        return result

    for module in (mcstastox, sampling, writer):
        monkeypatch.setattr(module, "sample_event_chunks", observed)
    return calls


def _forbidden(*args, **kwargs):
    pytest.fail("Metadata inspection or an empty projection executed user code or I/O")


def test_transform_specification_is_immutable_complete_and_identity_based():
    root = event_source(variables=("x", "y", "id")).sample(SamplingSettings(7))
    inputs = ["x", "y"]
    schema = {"sum": "float32", "difference": "int64"}
    projected = root.transform(
        _forbidden, inputs=inputs, schema=schema, description="Two-variable arithmetic"
    )
    independent = root.transform(_forbidden, inputs=inputs, schema=schema)
    inputs.reverse()
    schema["sum"] = "int8"
    chained = projected.transform(
        _forbidden, inputs=("sum", "difference"), schema={"score": "float64"}
    )
    column = chained["score"].scale(2).offset(3)

    assert isinstance(projected, EventTransform)
    assert not isinstance(projected, EventSource)
    assert not hasattr(projected, "sample")
    assert not hasattr(projected, "iter_chunks")
    assert projected.source is root
    assert chained.source is projected
    assert isinstance(column, EventColumn)
    assert column.source is chained
    assert root.execution_source is root
    assert projected.execution_source is chained.execution_source is root
    assert column.execution_source is root
    assert projected.sampling is chained.sampling is root.sampling
    assert len({projected, independent, chained}) == 3
    assert projected.output_variables == ("sum", "difference")
    assert chained.output_variables == ("score",)
    assert projected.operation.inputs == ("x", "y")
    assert projected.operation.schema == {
        "sum": np.dtype("float32"),
        "difference": np.dtype("int64"),
    }
    assert isinstance(projected.operation, ChunkOperation)
    assert chained.chunk_operations == (projected.operation, chained.operation)
    assert column.chunk_operations == chained.chunk_operations
    assert "Two-variable arithmetic" in projected.description
    assert "declared output schema" in repr(projected)
    with pytest.raises(FrozenInstanceError):
        projected.source = independent
    with pytest.raises(TypeError):
        projected.operation.schema["sum"] = np.dtype("float64")
    for name in ("x", "id", "p"):
        with pytest.raises(KeyError, match="not declared"):
            projected[name]


def test_multivariable_projection_receives_only_private_read_only_inputs():
    root = event_source(variables=("x", "y", "id", "t"))
    chunk = {
        "x": np.array([2.0, 4, 6]),
        "y": np.array([1.0, 2, 3]),
        "id": np.array([7.0, 8, 9]),
        "t": np.array([0.125, 0.25, 0.5]),
    }
    original = {name: values.copy() for name, values in chunk.items()}

    def calculate(private):
        assert isinstance(private, dict)
        assert tuple(private) == ("x", "y", "id")
        for name, values in private.items():
            assert not values.flags.writeable
            assert not np.shares_memory(values, chunk[name])
            with pytest.raises(ValueError, match="read-only"):
                values[0] = -99
        # Re-enabling writes to a private copy must still not affect a sibling.
        private["x"].flags.writeable = True
        private["x"][0] = 10
        return {
            "pixel": private["id"],
            "difference": private["x"] - private["y"],
            "sum": private["x"] + private["y"],
        }

    projected = root.transform(
        calculate,
        inputs=("x", "y", "id"),
        schema={"sum": "float32", "difference": "int16", "pixel": "uint32"},
    )
    result = projected.apply(chunk)
    assert tuple(result) == projected.output_variables
    for name, expected in (
        ("sum", [11, 6, 9]),
        ("difference", [9, 2, 3]),
        ("pixel", [7, 8, 9]),
    ):
        np.testing.assert_array_equal(result[name], expected)
        assert result[name].shape == (3,)
        assert result[name].dtype == projected.operation.schema[name]
    for name in chunk:
        np.testing.assert_array_equal(chunk[name], original[name])
        np.testing.assert_array_equal(root[name].apply(chunk), original[name])


@pytest.mark.parametrize("mode", ["unsampled", "sampled", "ordered"])
@pytest.mark.parametrize("bind_common", [False, True])
def test_branch_and_chain_share_one_execution_per_output_batch(
    source_file, tmp_path, sampler_calls, mode, bind_common
):
    before = source_file.read_bytes()
    root = event_source(
        component_name=("First", "Second"), variables=_VARIABLES, read_chunk_size=5
    )
    if mode != "unsampled":
        root = root.sample(
            SamplingSettings(23, seed=7, ordered=mode == "ordered", weight_key="w")
        )
    calls = {name: [] for name in ("common", "left", "right", "chain")}

    def common_calculation(chunk):
        assert len(sampler_calls) == (mode != "unsampled")
        calls["common"].append(len(chunk["id"]))
        return {
            "u": chunk["x"] + chunk["y"],
            "v": chunk["x"] - chunk["y"],
            "id": chunk["id"],
            "t": chunk["t"] - chunk["L"] / 8,
            "w": chunk["w"],
        }

    def left_calculation(chunk):
        assert tuple(chunk) == ("u", "v", "id", "t")
        calls["left"].append(len(chunk["id"]))
        return {
            "signal": 2 * chunk["u"] + chunk["v"],
            "pixel": chunk["id"] + 100,
            "tof": chunk["t"] + chunk["u"] / 16,
        }

    def right_calculation(chunk):
        calls["right"].append(len(chunk["id"]))
        return {"contrast": chunk["u"] - chunk["v"], "pixel": chunk["id"] + 200}

    def chain_calculation(chunk):
        assert chunk["pixel"].dtype == np.dtype("int32")
        calls["chain"].append(len(chunk["pixel"]))
        return {
            "score": chunk["signal"] + chunk["pixel"],
            "pixel": chunk["pixel"],
            "tof": chunk["tof"],
        }

    common = root.transform(
        common_calculation,
        inputs=("x", "y", "id", "t", "L", "w"),
        schema={
            "u": "float64",
            "v": "float64",
            "id": "int64",
            "t": "float64",
            "w": "float64",
        },
        description="Common geometry correction",
    )
    left = common.transform(
        left_calculation,
        inputs=("u", "v", "id", "t"),
        schema={"signal": "float64", "pixel": "int32", "tof": "float64"},
        description="Left branch",
    )
    right = common.transform(
        right_calculation,
        inputs=("u", "v", "id"),
        schema={"contrast": "float32", "pixel": "int32"},
        description="Right branch",
    )
    chain = left.transform(
        chain_calculation,
        inputs=("signal", "pixel", "tof"),
        schema={"score": "float64", "pixel": "int64", "tof": "float64"},
        description="Chained score",
    )
    converted = chain["tof"].convert_units("s", "ms").scale(2).offset(3)
    template = create_empty_template()
    template.add_group("/entry").add_group("events")
    template.add_group("/analysis").add_group("far")
    template.add_dataset("/entry/events/pixel").copy_from(left["pixel"])
    template.add_dataset("/entry/events/signal").copy_from(
        left["signal"].scale(2).offset(1)
    )
    template.add_dataset("/analysis/far/tof").copy_from(converted)
    template.add_dataset("/analysis/far/score").copy_from(chain["score"])
    template.add_dataset("/analysis/right").copy_from(
        right, columns=("pixel", "contrast")
    )
    template.add_dataset("/entry/events/table").copy_from(
        chain, columns=("tof", "pixel", "score"), write_chunk_size=2
    )
    template.add_dataset("/raw_id").copy_from(root["id"])
    template.add_dataset("/raw_time").copy_from(root["t"])
    if bind_common:
        template.add_dataset("/analysis/common").copy_from(
            common, columns=("w", "u", "id")
        )
        template.add_dataset("/common_v").copy_from(common["v"])
    results = []

    def select_count(result):
        results.append(result)
        return result.n_written

    template.add_dataset("/root_count").set_from_result(root, select_count)
    template.add_dataset("/alias_count").set_from_result(chain, select_count)
    template.attrs["branch_count"].set_from_result(right, select_count)
    output = tmp_path / "branches.nxs"
    report = write_nexus(template, output, source=source_file, write_chunk_size=7)

    assert set(report.sources) == set(report.preprocessing) == {root}
    assert report.preprocessing[root] == ()
    assert len(results) == 3
    assert all(result is report.sources[root] for result in results)
    if mode == "unsampled":
        assert sampler_calls == []
        rows = np.concatenate((_event_rows(range(9)), _event_rows(range(20, 26))))
        expected = dict(zip(_VARIABLES, rows.T, strict=True))
        lengths = (5, 4, 5, 1)
        assert report.sources[root].n_selected is None
    else:
        assert len(sampler_calls) == 1
        settings, expected = sampler_calls[0]
        assert settings is root.sampling
        np.testing.assert_array_equal(expected["w"], np.ones(23))
        assert report.sources[root].n_selected == 23
        assert report.sources[root].total_weight == expected.total_weight
        assert report.sources[root].effective_duration == expected.effective_duration
        lengths = (23,)
        if mode == "ordered":
            assert np.all(np.diff(expected["id"]) >= 0)
    expected_batches = [
        min(2, length - start) for length in lengths for start in range(0, length, 2)
    ]
    assert all(batches == expected_batches for batches in calls.values())
    count = len(expected["id"])
    assert report.sources[root].n_written == count
    np.testing.assert_array_equal(expected["t"], (expected["id"] + 1) / 8)
    u, v = expected["x"] + expected["y"], expected["x"] - expected["y"]
    pixel = expected["id"] + 100
    signal = 2 * u + v
    tof = expected["t"] - expected["L"] / 8 + u / 16
    score = signal + pixel
    expected_outputs = {
        "/entry/events/pixel": (pixel, "int32", left["pixel"]),
        "/entry/events/signal": (
            signal * 2 + 1,
            "float64",
            left["signal"].scale(2).offset(1),
        ),
        "/analysis/far/tof": (tof * 2000 + 3, "float64", converted),
        "/analysis/far/score": (score, "float64", chain["score"]),
        "/analysis/right": (
            np.column_stack((expected["id"] + 200, u - v)),
            "float64",
            right,
        ),
        "/entry/events/table": (np.column_stack((tof, pixel, score)), "float64", chain),
        "/raw_id": (expected["id"], "float64", root["id"]),
        "/raw_time": (expected["t"], "float64", root["t"]),
    }
    if bind_common:
        expected_outputs["/analysis/common"] = (
            np.column_stack((expected["w"], u, expected["id"])),
            "float64",
            common,
        )
        expected_outputs["/common_v"] = (v, "float64", common["v"])
    with h5py.File(output, "r") as handle:
        for path, (values, dtype, provider) in expected_outputs.items():
            dataset = handle[path]
            assert dataset.shape == values.shape
            assert dataset.dtype == np.dtype(dtype)
            np.testing.assert_array_equal(dataset[:], values)
            destination = report.destinations[path]
            assert destination.source is root
            assert destination.shape == values.shape
            assert destination.dtype == dataset.dtype
            assert destination.n_written == count
            stages = getattr(provider, "chunk_operations", ())
            assert isinstance(destination.chunk_operations, tuple)
            assert len(destination.chunk_operations) == len(stages)
            assert all(
                actual is stage
                for actual, stage in zip(
                    destination.chunk_operations, stages, strict=True
                )
            )
            column_operations = (
                provider.operations if isinstance(provider, EventColumn) else ()
            )
            assert destination.transformations == tuple(
                stage.description for stage in (*stages, *column_operations)
            )
        assert handle["root_count"][()] == handle["alias_count"][()] == count
        assert handle.attrs["branch_count"] == count
    assert source_file.read_bytes() == before


@pytest.mark.parametrize("sampled", [False, True])
def test_preprocessing_composes_with_post_sampling_multivariable_transforms(
    source_file, tmp_path, sampler_calls, sampled
):
    pre_batches, post_batches = [], []

    def accept_and_derive(chunk):
        pre_batches.append(len(chunk["id"]))
        keep = chunk["id"] % 2 == 0
        return {
            "w": chunk["w"][keep],
            "voxel": chunk["id"][keep] + 10,
            "t": (chunk["t"] + chunk["L"] / 16)[keep],
            "u": (chunk["x"] + chunk["y"])[keep],
        }

    def correct(chunk):
        assert tuple(chunk) == ("voxel", "t", "u", "w")
        assert chunk["voxel"].dtype == np.dtype("int32")
        assert len(sampler_calls) == sampled
        post_batches.append(len(chunk["voxel"]))
        return {
            "id": chunk["voxel"] * 2,
            "tof": chunk["t"] - chunk["u"] / 32,
            "w": chunk["w"],
        }

    root = event_source(
        component_name="First",
        variables=("id", "x", "y", "t", "L", "w"),
        read_chunk_size=4,
    ).preprocess(
        accept_and_derive,
        schema={"w": "float64", "voxel": "int32", "t": "float64", "u": "float64"},
        description="Even voxel acceptance",
    )
    if sampled:
        root = root.sample(SamplingSettings(17, seed=42, weight_key="w"))
    projected = root.transform(
        correct,
        inputs=("voxel", "t", "u", "w"),
        schema={"id": "int64", "tof": "float64", "w": "float64"},
        description="Accepted voxel correction",
    )
    template = create_empty_template()
    template.add_dataset("/table").copy_from(projected, columns=("w", "tof", "id"))
    template.add_dataset("/id").copy_from(projected["id"])
    template.add_dataset("/tof").copy_from(projected["tof"].convert_units("s", "ms"))
    template.add_dataset("/count").set_from_result(projected, "n_written")
    output = tmp_path / "prepared.nxs"
    report = write_nexus(template, output, source=source_file, write_chunk_size=2)

    assert pre_batches == [4, 4, 1]
    assert set(report.sources) == set(report.preprocessing) == {root}
    assert report.preprocessing[root] is root.preprocessors
    assert len(sampler_calls) == sampled
    if sampled:
        selected = sampler_calls[0][1]
        assert set(selected) == {"w", "voxel", "t", "u"}
        assert report.sources[root].total_weight == 10
        assert report.sources[root].total_weight_variance == 30
        assert report.sources[root].effective_duration == pytest.approx(17 / 10)
        np.testing.assert_array_equal(selected["w"], np.ones(17))
        ids = selected["voxel"] * 2
        tof = selected["t"] - selected["u"] / 32
        weights = selected["w"]
        assert post_batches == [2] * 8 + [1]
    else:
        rows = _event_rows(range(0, 9, 2))
        ids = (rows[:, 2] + 10) * 2
        tof = rows[:, 1] + rows[:, 5] / 16 - (rows[:, 3] + rows[:, 4]) / 32
        weights = rows[:, 6]
        assert post_batches == [2, 2, 1]
    assert sum(post_batches) == len(ids)
    original_ids = ids / 2 - 10
    assert set(original_ids) <= {0, 2, 4, 6, 8}
    np.testing.assert_array_equal(
        tof,
        (original_ids + 1) / 8
        + (original_ids + 2) / 16
        - (original_ids - 3 + original_ids % 3 - 1) / 32,
    )
    with h5py.File(output, "r") as handle:
        np.testing.assert_array_equal(
            handle["table"][:], np.column_stack((weights, tof, ids))
        )
        np.testing.assert_array_equal(handle["id"][:], ids)
        np.testing.assert_array_equal(handle["tof"][:], tof * 1000)
        assert handle["id"].dtype == np.dtype("int64")
        assert handle["count"][()] == report.sources[root].n_written == len(ids)
    for path in ("/table", "/id", "/tof"):
        assert report.destinations[path].source is root
        assert report.destinations[path].chunk_operations == (projected.operation,)


@pytest.mark.parametrize(
    ("component", "sampled"), [("Empty", False), ("Empty", True), ("Zero", True)]
)
def test_empty_outputs_use_declared_schemas_without_callback_probes(
    source_file, tmp_path, sampler_calls, component, sampled
):
    root = event_source(
        component_name=component, variables=("id", "t"), read_chunk_size=2
    )
    if sampled:
        root = root.sample(SamplingSettings(11, seed=7, weight_key="w"))
    projected = root.transform(
        _forbidden, inputs=("id", "t"), schema={"pixel": "int32", "tof": "float32"}
    )
    chain = projected.transform(
        _forbidden, inputs=("pixel",), schema={"score": "int64"}
    )
    template = create_empty_template()
    template.add_dataset("/id").copy_from(projected["pixel"])
    template.add_dataset("/tof").copy_from(projected["tof"])
    template.add_dataset("/table").copy_from(projected, columns=("tof", "pixel"))
    template.add_dataset("/score").copy_from(
        chain["score"].map(_forbidden, dtype="int16")
    )
    output = tmp_path / "empty.nxs"
    report = write_nexus(template, output, source=source_file, write_chunk_size=2)

    assert len(sampler_calls) == sampled
    assert set(report.sources) == {root}
    assert report.sources[root].n_written == 0
    assert report.sources[root].n_selected == (0 if sampled else None)
    if sampled:
        assert report.sources[root].total_weight == 0
        assert report.sources[root].effective_duration is None
    with h5py.File(output, "r") as handle:
        for path, shape, dtype in (
            ("/id", (0,), "int32"),
            ("/tof", (0,), "float32"),
            ("/table", (0, 2), "float64"),
            ("/score", (0,), "int16"),
        ):
            assert handle[path].shape == shape
            assert handle[path].dtype == np.dtype(dtype)
            assert report.destinations[path].source is root
            assert report.destinations[path].n_written == 0


@pytest.mark.parametrize("mode", ["unsampled", "seeded", "unseeded"])
def test_fresh_writes_are_bounded_and_independent_of_write_batch_size(
    source_file, tmp_path, monkeypatch, sampler_calls, mode
):
    root = event_source(
        component_name="First", variables=("id", "t", "x", "y"), read_chunk_size=5
    )
    if mode != "unsampled":
        root = root.sample(SamplingSettings(19, seed=7 if mode == "seeded" else None))
    callback_batches, writes = [], []
    limit = 0

    def derive(chunk):
        size = len(chunk["id"])
        assert 0 < size <= limit
        callback_batches.append(size)
        return {
            "id": chunk["id"],
            "corrected": chunk["t"] + (chunk["x"] ** 2 + chunk["y"] ** 2) / 32,
        }

    projected = root.transform(
        derive,
        inputs=("id", "t", "x", "y"),
        schema={"id": "int32", "corrected": "float64"},
    )
    template = create_empty_template()
    template.add_dataset("/id").copy_from(projected["id"])
    template.add_dataset("/table").copy_from(projected, columns=("corrected", "id"))
    before = template.summary()
    setitem = h5py.Dataset.__setitem__

    def bounded_write(dataset, selection, values):
        if dataset.name in {"/id", "/table"}:
            size = np.asarray(values).shape[0]
            assert 0 < size <= limit
            writes.append((dataset.name, size))
        return setitem(dataset, selection, values)

    monkeypatch.setattr(h5py.Dataset, "__setitem__", bounded_write)
    outputs, reports = [], []
    for number, limit in enumerate((1, 4)):
        callback_batches.clear()
        writes.clear()
        output = tmp_path / f"fresh-{number}.nxs"
        report = write_nexus(
            template, output, source=source_file, write_chunk_size=limit
        )
        reports.append(report)
        assert template.summary() == before
        expected_count = 9 if mode == "unsampled" else 19
        assert sum(callback_batches) == expected_count
        for path in ("/id", "/table"):
            assert [size for name, size in writes if name == path] == callback_batches
        with h5py.File(output, "r") as handle:
            ids, table = handle["id"][:], handle["table"][:]
            outputs.append((ids, table))
            expected_ids = (
                np.arange(9) if mode == "unsampled" else sampler_calls[number][1]["id"]
            )
            np.testing.assert_array_equal(ids, expected_ids)
            np.testing.assert_array_equal(table[:, 1], ids)
            np.testing.assert_array_equal(
                table[:, 0], (ids + 1) / 8 + ((ids - 3) ** 2 + (ids % 3 - 1) ** 2) / 32
            )
        assert report.sources[root].n_written == expected_count
        assert len(sampler_calls) == (0 if mode == "unsampled" else number + 1)
    assert reports[0].sources[root] is not reports[1].sources[root]
    if mode != "unsampled":
        assert sampler_calls[0][1] is not sampler_calls[1][1]
    if mode != "unseeded":
        for first, second in zip(outputs[0], outputs[1], strict=True):
            np.testing.assert_array_equal(first, second)


def test_unsampled_transform_streams_without_collecting_future_input(
    source_file, tmp_path, monkeypatch, sampler_calls
):
    processed = 0
    iterations = []
    original = McStasNeXus.iter_event_data

    def observed(reader, variables, *, component_name=None, chunk_size):
        iterations.append(chunk_size)
        delivered = 0
        for chunk in original(
            reader, variables, component_name=component_name, chunk_size=chunk_size
        ):
            delivered += len(chunk["id"])
            yield chunk
            assert processed == delivered, (
                "Writer collected input before transforming it"
            )

    def derive(chunk):
        nonlocal processed
        assert len(chunk["id"]) <= 2
        processed += len(chunk["id"])
        return {"pixel": chunk["id"] + chunk["x"], "t": chunk["t"]}

    monkeypatch.setattr(McStasNeXus, "iter_event_data", observed)
    monkeypatch.setattr(McStasNeXus, "get_event_data", _forbidden)
    root = event_source(
        component_name=("First", "Second"),
        variables=("id", "x", "t"),
        read_chunk_size=4,
    )
    projected = root.transform(
        derive, inputs=("id", "x", "t"), schema={"pixel": "int64", "t": "float64"}
    )
    template = create_empty_template()
    template.add_dataset("/pixel").copy_from(projected["pixel"])
    template.add_dataset("/time").copy_from(projected["t"])
    report = write_nexus(
        template, tmp_path / "stream.nxs", source=source_file, write_chunk_size=2
    )

    assert iterations == [4]
    assert processed == report.sources[root].n_written == 15
    assert sampler_calls == []


def test_summary_and_validation_are_metadata_only_and_share_root_sample_labels(
    source_file, monkeypatch
):
    root = (
        event_source(component_name="First", variables=("x", "y", "t", "w"))
        .preprocess(
            _forbidden,
            schema={"x": "float64", "y": "float64", "t": "float64", "w": "float64"},
            description="Input preparation",
        )
        .sample(SamplingSettings(13, seed=42, weight_key="w"))
    )
    projected = root.transform(
        _forbidden,
        inputs=("x", "y", "t"),
        schema={"pixel": "int32", "tof": "float64"},
        description="Geometry projection",
    )
    chain = projected.transform(
        _forbidden,
        inputs=("pixel", "tof"),
        schema={"score": "float32"},
        description="Combined score",
    )
    branch = projected.transform(
        _forbidden, inputs=("pixel",), schema={"other": "int64"}
    )
    template = create_empty_template()
    template.add_dataset("/raw_time").copy_from(root["t"])
    template.add_dataset("/pixel").copy_from(
        projected["pixel"].map(_forbidden, dtype="int64")
    )
    template.add_dataset("/chain").copy_from(chain, columns=("score",))
    template.add_dataset("/branch").copy_from(branch["other"])
    template.add_dataset("/duration").set_from_result(chain, "effective_duration")
    template.add_dataset("/count").set_from_result(root, _forbidden)
    template.add_dataset("/title").set_from(_forbidden, description="Source callback")
    template.attrs["source_attribute"].set_from(_forbidden)
    template.attrs["result_attribute"].set_from_result(branch, _forbidden)
    with SourceContext(source_file) as context:
        for name in ("__getitem__", "__array__", "read_direct"):
            monkeypatch.setattr(h5py.Dataset, name, _forbidden)
        for name in ("iter_chunks", "schema", "raw_schema", "row_count"):
            monkeypatch.setattr(EventSource, name, _forbidden)
        monkeypatch.setattr(EventTransform, "apply", _forbidden)
        monkeypatch.setattr(EventColumn, "apply", _forbidden)
        monkeypatch.setattr(
            importlib.import_module("mcstastox.NexusWrite"),
            "sample_event_chunks",
            _forbidden,
        )
        template.validate(source=context).raise_if_errors()
        text = template.summary(source=context)
        assert [
            line.split(":", 1)[0]
            for line in text.splitlines()
            if line.startswith("sample")
        ] == ["sample1"]
        assert text.count("sample1 ->") == 4
        assert "sample2" not in text
        assert "source1" not in text
        assert "sample1 result -> effective_duration" in text
        assert "raw read dependencies" in text
        assert "Input preparation" in text
        assert "chunk transform 1: inputs=[x, y, t]" in text
        assert "chunk transform 2: inputs=[pixel, tof]" in text
        assert "Geometry projection" in text
        assert "Combined score" in text
        assert "declared output schema: {pixel: int32, tof: float64}" in text
        assert "declared output schema: {score: float32}" in text
        assert "Source callback" in text
        assert "<function" not in text
        assert "0x" not in text
        assert "EventTransform" in repr(chain)
        assert "EventColumn" in repr(chain["score"])
        bounded = template.summary(source=object(), limit=2)
        assert "more nodes" in bounded
        assert len(bounded) < 3000


@pytest.mark.parametrize("sampled", [False, True])
def test_result_only_transform_alias_uses_root_statistics_without_projection(
    source_file, tmp_path, sampler_calls, sampled
):
    root = event_source(component_name="First", variables=("id",))
    if sampled:
        root = root.sample(SamplingSettings(7, seed=42))
    alias = root.transform(_forbidden, inputs=("id",), schema={"pixel": "int64"})
    selected = []

    def count(result):
        selected.append(result)
        return result.n_written

    template = create_empty_template()
    template.add_dataset("/count").set_from_result(alias, count)
    template.attrs["count"].set_from_result(root, count)
    output = tmp_path / "statistics.nxs"
    report = write_nexus(template, output, source=source_file)

    assert set(report.sources) == set(report.preprocessing) == {root}
    assert len(selected) == 2
    assert all(result is report.sources[root] for result in selected)
    assert len(sampler_calls) == sampled
    with h5py.File(output, "r") as handle:
        assert handle["count"][()] == handle.attrs["count"] == (7 if sampled else 9)


@pytest.mark.parametrize("missing", ["inputs", "schema"])
def test_transform_requires_explicit_inputs_and_schema(missing):
    kwargs = {"inputs": ("x",), "schema": {"out": "float64"}}
    del kwargs[missing]
    with pytest.raises(TypeError, match=missing):
        event_source(variables=("x",)).transform(_forbidden, **kwargs)


@pytest.mark.parametrize(
    "invalid",
    [
        pytest.param({"callback": None}, id="noncallable"),
        pytest.param({"inputs": None}, id="unspecified-inputs"),
        pytest.param({"inputs": ()}, id="empty-inputs"),
        pytest.param({"inputs": ("x", "x")}, id="duplicate-inputs"),
        pytest.param({"inputs": ("",)}, id="empty-input-name"),
        pytest.param({"inputs": ("x y",)}, id="whitespace-input"),
        pytest.param({"inputs": (1,)}, id="nonstr-input"),
        pytest.param({"inputs": {"x"}}, id="unordered-inputs"),
        pytest.param({"schema": []}, id="nonmapping-schema"),
        pytest.param({"schema": {}}, id="empty-schema"),
        pytest.param({"schema": {"": "int64"}}, id="empty-output-name"),
        pytest.param({"schema": {"bad name": "int64"}}, id="whitespace-output"),
        pytest.param({"schema": {1: "int64"}}, id="nonstr-output"),
        pytest.param({"schema": {"out": "bool"}}, id="bool-schema"),
        pytest.param({"schema": {"out": "S4"}}, id="text-schema"),
        pytest.param({"schema": {"out": "object"}}, id="object-schema"),
        pytest.param({"schema": {"out": "datetime64[s]"}}, id="datetime-schema"),
        pytest.param({"description": 12}, id="nonstr-description"),
    ],
)
def test_invalid_transform_declarations_fail_without_executing_callbacks(invalid):
    kwargs = {
        "callback": _forbidden,
        "inputs": ("x", "y"),
        "schema": {"out": "float64"},
    }
    kwargs.update(invalid)
    with pytest.raises((TypeError, ValueError)):
        event_source(variables=("x", "y")).transform(**kwargs)


def test_transform_inputs_are_exact_declared_names_with_no_implicit_passthrough():
    root = event_source(variables=("id", "t")).sample(
        SamplingSettings(7, weight_key="w")
    )
    for name in ("missing", "identity", "w"):
        with pytest.raises(KeyError, match="not declared"):
            root.transform(_forbidden, inputs=(name,), schema={"out": "float64"})
    first = root.transform(_forbidden, inputs=("id",), schema={"pixel": "int64"})
    with pytest.raises(KeyError, match="not declared"):
        first["t"]
    with pytest.raises(KeyError, match="not declared"):
        first.transform(_forbidden, inputs=("id",), schema={"next": "int64"})
    prepared = event_source(variables=("x",)).preprocess(
        _forbidden, schema={"voxel": "int64"}
    )
    with pytest.raises(KeyError, match="not declared"):
        prepared.transform(_forbidden, inputs=("x",), schema={"out": "int64"})
    assert (
        prepared.transform(
            _forbidden, inputs=("voxel",), schema={"out": "int64"}
        ).execution_source
        is prepared
    )


@pytest.mark.parametrize("bad_source", [True, False])
def test_direct_event_transform_requires_a_source_and_declared_chunk_operation(
    bad_source,
):
    root = event_source(variables=("x",))
    operation = ChunkOperation(_forbidden, {"out": "float64"}, inputs=("x",))
    with pytest.raises(TypeError):
        EventTransform(
            object() if bad_source else root, operation if bad_source else object()
        )
    with pytest.raises(TypeError, match="declared input"):
        EventTransform(root, ChunkOperation(_forbidden, {"out": "float64"}))


@pytest.mark.parametrize(
    ("result", "dtype", "error", "message"),
    [
        pytest.param(None, "float64", TypeError, "dictionary", id="none"),
        pytest.param([0, 1, 2], "float64", TypeError, "dictionary", id="list"),
        pytest.param(np.arange(3), "float64", TypeError, "dictionary", id="array"),
        pytest.param({}, "float64", ValueError, "keys", id="empty-keys"),
        pytest.param({"u": [0, 1, 2]}, "float64", ValueError, "keys", id="missing-key"),
        pytest.param(
            {"u": [0, 1, 2], "v": [0, 1, 2], "extra": [0, 1, 2]},
            "float64",
            ValueError,
            "keys",
            id="extra-key",
        ),
        pytest.param(
            {"u": [0, 1, 2], "other": [0, 1, 2]},
            "float64",
            ValueError,
            "keys",
            id="wrong-key",
        ),
        pytest.param(
            {"u": 1, "v": [0, 1, 2]},
            "float64",
            ValueError,
            "one-dimensional",
            id="scalar",
        ),
        pytest.param(
            {"u": [[0], [1], [2]], "v": [0, 1, 2]},
            "float64",
            ValueError,
            "one-dimensional",
            id="rank-two",
        ),
        pytest.param(
            {"u": [0, 1], "v": [0, 1]},
            "float64",
            ValueError,
            "preserve.*row count",
            id="shrinking",
        ),
        pytest.param(
            {"u": [0, 1, 2, 3], "v": [0, 1, 2, 3]},
            "float64",
            ValueError,
            "preserve.*row count",
            id="expanding",
        ),
        pytest.param(
            {"u": [0, 1, 2], "v": [0, 1]},
            "float64",
            ValueError,
            "same row count",
            id="unaligned",
        ),
        pytest.param(
            {"u": [0.5, 1, 2], "v": [0, 1, 2]},
            "int64",
            ValueError,
            "integral",
            id="fractional",
        ),
        pytest.param(
            {"u": [0, 128, 2], "v": [0, 1, 2]},
            "int8",
            OverflowError,
            "range",
            id="signed-overflow",
        ),
        pytest.param(
            {"u": [0, -1, 2], "v": [0, 1, 2]},
            "uint64",
            OverflowError,
            "range",
            id="negative-unsigned",
        ),
        pytest.param(
            {"u": [0, float(2**63), 2], "v": [0, 1, 2]},
            "int64",
            OverflowError,
            "range",
            id="int64-boundary",
        ),
        pytest.param(
            {"u": np.array([0, 2**64, 2], dtype=object), "v": [0, 1, 2]},
            "uint64",
            OverflowError,
            "range",
            id="uint64-boundary",
        ),
        pytest.param(
            {"u": [0, np.nan, 2], "v": [0, 1, 2]},
            "int64",
            ValueError,
            "finite integral",
            id="nan-integer",
        ),
        pytest.param(
            {"u": [0, np.inf, 2], "v": [0, 1, 2]},
            "int64",
            ValueError,
            "finite integral",
            id="infinite-integer",
        ),
        pytest.param(
            {"u": [0, 1e100, 2], "v": [0, 1, 2]},
            "float32",
            OverflowError,
            "overflow",
            id="float-overflow",
        ),
        pytest.param(
            {"u": ["0", "1", "2"], "v": [0, 1, 2]},
            "float64",
            TypeError,
            "numeric",
            id="numeric-text",
        ),
        pytest.param(
            {"u": np.array([0, object(), 2], dtype=object), "v": [0, 1, 2]},
            "float64",
            TypeError,
            "numeric",
            id="nonnumeric-object",
        ),
        pytest.param(
            {"u": [True, False, True], "v": [0, 1, 2]},
            "float64",
            TypeError,
            "numeric",
            id="boolean-result",
        ),
        pytest.param(
            {"u": [0, 1 + 2j, 2], "v": [0, 1, 2]},
            "float64",
            TypeError,
            "complex",
            id="complex-to-real",
        ),
        pytest.param(
            {"u": np.array([0, np.complex64(1 + 2j), 2], dtype=object), "v": [0, 1, 2]},
            "float64",
            TypeError,
            "complex",
            id="complex-object-to-real",
        ),
    ],
)
def test_invalid_callback_results_fail_atomically_even_for_unselected_outputs(
    source_file, tmp_path, result, dtype, error, message
):
    before = source_file.read_bytes()
    root = event_source(component_name="First", variables=("id",), read_chunk_size=5)
    projected = root.transform(
        lambda chunk: result, inputs=("id",), schema={"u": dtype, "v": "int64"}
    )
    template = create_empty_template()
    template.add_dataset("/raw").copy_from(root["id"])
    template.add_dataset("/projected").copy_from(projected["v"])
    output = tmp_path / "invalid.nxs"
    siblings = set(tmp_path.iterdir())
    with pytest.raises(error, match=message) as caught:
        write_nexus(template, output, source=source_file, write_chunk_size=3)
    assert "/projected" in str(caught.value)
    assert not output.exists()
    assert set(tmp_path.iterdir()) == siblings
    assert source_file.read_bytes() == before


def test_explicit_numeric_output_casts_accept_integral_lists_and_complex_schemas():
    projected = event_source(variables=("id",)).transform(
        lambda chunk: {"pixel": [2.0, 1.0], "amplitude": [1 + 2j, 3 - 4j]},
        inputs=("id",),
        schema={"pixel": "uint16", "amplitude": "complex64"},
    )
    result = projected.apply({"id": np.array([0, 1])})
    assert result["pixel"].dtype == np.dtype("uint16")
    assert result["amplitude"].dtype == np.dtype("complex64")
    np.testing.assert_array_equal(result["pixel"], [2, 1])
    np.testing.assert_array_equal(result["amplitude"], [1 + 2j, 3 - 4j])


@pytest.mark.parametrize("weight_key", ["p", "w"])
@pytest.mark.parametrize("bad_weight", [0, 2, np.nan, np.inf])
def test_sampled_weight_outputs_cannot_be_reintroduced_with_nonunit_values(
    source_file, tmp_path, sampler_calls, weight_key, bad_weight
):
    root = event_source(component_name="First", variables=("id",)).sample(
        SamplingSettings(7, seed=42, weight_key=weight_key)
    )

    def change_weight(chunk):
        weights = np.ones(len(chunk["id"]))
        weights[-1] = bad_weight
        return {"id": chunk["id"], weight_key: weights}

    projected = root.transform(
        change_weight, inputs=("id",), schema={"id": "int64", weight_key: "float64"}
    )
    template = create_empty_template()
    template.add_dataset("/id").copy_from(projected["id"])
    output = tmp_path / "bad-weight.nxs"
    siblings = set(tmp_path.iterdir())
    with pytest.raises(ValueError, match="unit weights"):
        write_nexus(template, output, source=source_file, write_chunk_size=3)
    assert len(sampler_calls) == 1
    assert set(tmp_path.iterdir()) == siblings


@pytest.mark.parametrize(
    "modify",
    [
        pytest.param(lambda column: column.scale(2), id="scale"),
        pytest.param(lambda column: column.offset(1), id="offset"),
        pytest.param(
            lambda column: column.convert_units("s", "ms"), id="unit-conversion"
        ),
        pytest.param(
            lambda column: column.map_values({1: 1}, dtype="int64"), id="value-map"
        ),
        pytest.param(
            lambda column: column.map(lambda values: values, dtype="float64"),
            id="callback-map",
        ),
    ],
)
def test_subsequent_column_operations_cannot_change_transformed_sampled_weights(modify):
    root = event_source(variables=("id",)).sample(SamplingSettings(7, weight_key="w"))
    projected = root.transform(
        _forbidden, inputs=("id",), schema={"w": "float64", "other": "float64"}
    )
    chain = projected.transform(
        _forbidden, inputs=("w", "other"), schema={"w": "int64", "other": "float64"}
    )
    for provider in (projected, chain):
        with pytest.raises(ValueError, match="unit weights"):
            modify(provider["w"])
        assert modify(provider["other"]).execution_source is root


def test_sampled_unit_weight_outputs_and_identity_column_operations_are_allowed(
    source_file, tmp_path
):
    root = event_source(component_name="First", variables=("id",)).sample(
        SamplingSettings(7, seed=42, weight_key="w")
    )
    projected = root.transform(
        lambda chunk: {"id": chunk["id"], "w": np.ones(len(chunk["id"]))},
        inputs=("id",),
        schema={"id": "int64", "w": "int8"},
    )
    template = create_empty_template()
    template.add_dataset("/weights").copy_from(
        projected["w"].scale(1).offset(0).convert_units("s", "s", dtype="int16")
    )
    template.add_dataset("/table").copy_from(projected, columns=("w", "id"))
    output = tmp_path / "unit-weights.nxs"
    report = write_nexus(template, output, source=source_file, write_chunk_size=2)
    with h5py.File(output, "r") as handle:
        np.testing.assert_array_equal(handle["weights"][:], np.ones(7))
        assert handle["weights"].dtype == np.dtype("int16")
        np.testing.assert_array_equal(handle["table"][:, 0], np.ones(7))
    assert report.destinations["/weights"].source is root
    with pytest.raises(TypeError, match="real unit weights"):
        root.transform(_forbidden, inputs=("id",), schema={"w": "complex128"})


def test_unsampled_weight_columns_are_not_constrained_to_unit_weights():
    root = event_source(variables=("w",))
    projected = root.transform(
        lambda chunk: {"w": chunk["w"] * 2}, inputs=("w",), schema={"w": "float64"}
    )
    np.testing.assert_array_equal(
        projected["w"].offset(1).apply({"w": np.array([0.0, 2, 3])}), [1, 5, 7]
    )


def test_row_order_and_determinism_remain_caller_responsibilities():
    calls = 0

    def caller_defined(chunk):
        nonlocal calls
        calls += 1
        return {"id": chunk["id"][::-1] + calls}

    projected = event_source(variables=("id",)).transform(
        caller_defined, inputs=("id",), schema={"id": "int64"}
    )
    chunk = {"id": np.array([0, 1, 2])}
    np.testing.assert_array_equal(projected.apply(chunk)["id"], [3, 2, 1])
    np.testing.assert_array_equal(projected.apply(chunk)["id"], [4, 3, 2])
    np.testing.assert_array_equal(chunk["id"], [0, 1, 2])


@pytest.mark.parametrize("stage", ["preprocess", "transform"])
@pytest.mark.parametrize("pixel", [2**53 + 1, 2**64 - 1])
def test_mixed_integer_tables_preserve_ids_before_homogeneous_assembly(
    source_file, tmp_path, stage, pixel
):
    def prepare(chunk):
        count = len(chunk["id"])
        return {
            "pixel": np.full(count, pixel, dtype="uint64"),
            "signed": np.ones(count, dtype="int64"),
        }

    root = event_source(component_name="First", variables="id", read_chunk_size=2)
    kwargs = {"inputs": ("id",)} if stage == "transform" else {}
    events = getattr(root, stage)(
        prepare, schema={"pixel": "uint64", "signed": "int64"}, **kwargs
    )
    template = create_empty_template()
    template.add_dataset("/pixel").copy_from(events["pixel"])
    template.add_dataset("/table").copy_from(
        events, columns=("pixel", "signed"), dtype="uint64"
    )
    output = tmp_path / "large-ids.nxs"
    write_nexus(template, output, source=source_file, write_chunk_size=1)
    with h5py.File(output, "r") as handle:
        np.testing.assert_array_equal(handle["table"][:, 0], handle["pixel"][:])
        assert all(int(value) == pixel for value in handle["table"][:, 0])
        np.testing.assert_array_equal(handle["table"][:, 1], np.ones(9, dtype="uint64"))
