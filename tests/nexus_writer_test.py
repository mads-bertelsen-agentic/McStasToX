# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 McCode-dev contributors (https://github.com/mccode-dev)
"""Public writer integration tests using real, independent synthetic HDF5 inputs."""

import importlib
import math
from pathlib import Path

import h5py
import numpy as np
import pytest

import mcstastox
from mcstastox.ReadNeXus import McStasNeXus

_EVENTS = np.array(
    [
        [0, 0.000, 0, 1, 1.0],
        [1, 0.001, 1, 0, 1.5],
        [2, 0.002, 2, 2, 2.0],
        [0, 0.003, 3, 3, 2.5],
        [4, 0.004, 4, 4, 3.0],
    ],
    dtype="float64",
)


def _write_source(path):
    with h5py.File(path, "w") as file:
        for number in (1, 2, 10):
            entry = file.create_group(f"entry{number}")
            entry.create_group("data")
            simulation = entry.create_group("simulation")
            simulation.attrs["program"] = np.bytes_("3.8.6")
            simulation.create_group("Param")
            entry.create_dataset("title", data=np.bytes_(f"00{number}"))
            entry.create_dataset("scalar", data=np.int32(number))
            entry.create_dataset("empty", shape=(0, 3), dtype="int16")
            entry.create_dataset(
                "text", data=["001.50", "caf\u00e9"], dtype=h5py.string_dtype()
            )
            raw = entry.create_dataset(
                "raw", data=np.arange(23, dtype="int16") + number * 100, chunks=(4,)
            )
            raw.attrs["units"] = "source-only units"
            components = entry.create_group("instrument/components")
            for index, name in enumerate(("Detector", "Second", "Empty", "Zero")):
                rows = np.empty((0, 5)) if name == "Empty" else _EVENTS.copy()
                rows[:, 2] += (number - 1) * 100 + (10 if name == "Second" else 0)
                rows[:, 1] = rows[:, 2] * 0.001
                if name == "Zero":
                    rows[:, (0, 3)] = 0
                output = components.create_group(f"{index:04d}_{name}/output")
                output.create_group("BINS")
                info = output.create_group("event_data")
                info.attrs["variables"] = np.bytes_("p t id w L")
                info.create_dataset("events", data=rows)
    return path


@pytest.fixture
def source_file(tmp_path):
    return _write_source(tmp_path / "source.nxs")


@pytest.fixture
def base_file(tmp_path):
    path = tmp_path / "base.nxs"
    with h5py.File(path, "w") as file:
        file.attrs["creator"] = "facility"
        file.attrs["labels"] = np.array(
            ["alpha", "caf\u00e9"], dtype=h5py.string_dtype()
        )
        entry = file.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"
        data = entry.create_dataset(
            "data",
            data=np.arange(21, dtype="int32").reshape(7, 3),
            chunks=(3, 3),
            maxshape=(None, 3),
            compression="gzip",
            compression_opts=2,
            shuffle=True,
            fletcher32=True,
            fillvalue=-7,
            track_order=True,
        )
        data.attrs["units"] = "counts"
        data.attrs["unknown_metadata"] = np.array([b"001", b"002"], dtype="S3")
        entry.create_dataset("title", data=np.bytes_("old"))
        fixed = entry.create_dataset("fixed", data=np.array([b"old"], dtype="S3"))
        fixed.attrs["description"] = "retained text metadata"
        entry.create_dataset(
            "text", data=["old", "caf\u00e9"], dtype=h5py.string_dtype()
        )
        entry.create_dataset("empty", shape=(0, 3), dtype="int16")
        entry.create_dataset("obsolete/events", data=np.arange(45).reshape(5, 9))
        file.create_group("other")["alias"] = data
        entry["soft"] = h5py.SoftLink("data")
        file["entry_alias"] = entry
    return path


@pytest.fixture
def output_path(tmp_path):
    directory = tmp_path / "published"
    directory.mkdir()
    return directory / "written.nxs"


def _selection_size(shape, selection):
    if selection is None:
        return math.prod(shape)
    items = selection if isinstance(selection, tuple) else (selection,)
    missing = len(shape) - sum(item is not Ellipsis for item in items)
    expanded = []
    for item in items:
        expanded.extend([slice(None)] * missing if item is Ellipsis else [item])
    expanded.extend([slice(None)] * (len(shape) - len(expanded)))
    assert len(expanded) == len(shape)
    size = 1
    for dimension, item in zip(shape, expanded, strict=True):
        if isinstance(item, slice):
            size *= len(range(*item.indices(dimension)))
        else:
            assert isinstance(item, (int, np.integer))
    return size


def _watch_io(monkeypatch, *, read_limits, output_path, write_limits=None):
    """Observe HDF5 transfers, including read_direct/write_direct alternatives."""
    reads, writes = {}, {}
    write_limits = {} if write_limits is None else write_limits
    getitem = h5py.Dataset.__getitem__
    array = h5py.Dataset.__array__
    read_direct = h5py.Dataset.read_direct
    setitem = h5py.Dataset.__setitem__
    write_direct = h5py.Dataset.write_direct

    def record_read(dataset, selection):
        key = (Path(dataset.file.filename).resolve(), dataset.name)
        if key in read_limits:
            size = _selection_size(dataset.shape, selection)
            assert size <= read_limits[key], f"Unbounded read of {dataset.name}: {size}"
            reads.setdefault(dataset.name, []).append(size)

    def record_write(dataset, selection):
        if (
            Path(dataset.file.filename).parent == output_path.parent
            and dataset.name in write_limits
        ):
            size = _selection_size(dataset.shape, selection)
            assert size <= write_limits[dataset.name], (
                f"Unbounded write of {dataset.name}: {size}"
            )
            writes.setdefault(dataset.name, []).append(size)

    def guarded_getitem(dataset, selection, *args, **kwargs):
        record_read(dataset, selection)
        return getitem(dataset, selection, *args, **kwargs)

    def guarded_array(dataset, *args, **kwargs):
        key = (Path(dataset.file.filename).resolve(), dataset.name)
        assert key not in read_limits, f"Eager dataset conversion: {dataset.name}"
        return array(dataset, *args, **kwargs)

    def guarded_read_direct(dataset, dest, source_sel=None, dest_sel=None):
        record_read(dataset, source_sel)
        return read_direct(dataset, dest, source_sel=source_sel, dest_sel=dest_sel)

    def guarded_setitem(dataset, selection, value):
        record_write(dataset, selection)
        return setitem(dataset, selection, value)

    def guarded_write_direct(dataset, source, source_sel=None, dest_sel=None):
        record_write(dataset, dest_sel)
        return write_direct(dataset, source, source_sel=source_sel, dest_sel=dest_sel)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", guarded_getitem)
    monkeypatch.setattr(h5py.Dataset, "__array__", guarded_array)
    monkeypatch.setattr(h5py.Dataset, "read_direct", guarded_read_direct)
    monkeypatch.setattr(h5py.Dataset, "__setitem__", guarded_setitem)
    monkeypatch.setattr(h5py.Dataset, "write_direct", guarded_write_direct)
    return reads, writes


def _spy_sampler(monkeypatch):
    """Wrap the real public sampler, also covering a writer's imported alias."""
    sampling = importlib.import_module("mcstastox.Sampling")
    writer = importlib.import_module("mcstastox.NexusWrite")
    original = sampling.sample_event_chunks
    calls = []

    def observed(chunks, settings):
        result = original(chunks, settings)
        calls.append((settings, result))
        return result

    for module in (mcstastox, sampling, writer):
        if getattr(module, "sample_event_chunks", None) is original:
            monkeypatch.setattr(module, "sample_event_chunks", observed)
    return calls


def test_empty_template_writes_literals_attributes_scalars_and_empty_arrays(
    output_path,
):
    template = mcstastox.create_empty_template()
    template.attrs["creator"] = "integration test"
    entry = template.add_group("/entry")
    entry.attrs["NX_class"] = "NXentry"
    entry.add_dataset("title", value="Simulated events")
    entry.add_dataset("scalar", value=np.int32(42))
    data = entry.add_dataset("values", value=np.arange(6).reshape(2, 3))
    data.attrs["units"] = "explicit units"
    data.attrs["labels"] = ["first", "second"]
    entry.add_dataset("empty", shape=(0, 3), dtype="float32")

    report = mcstastox.write_nexus(template, output_path)

    with h5py.File(output_path, "r") as file:
        assert set(file) == {"entry"}
        assert file.attrs["creator"] == "integration test"
        assert file["entry"].attrs["NX_class"] == "NXentry"
        assert file["entry/title"].shape == ()
        assert file["entry/title"].asstr()[()] == "Simulated events"
        assert file["entry/scalar"].shape == ()
        assert file["entry/scalar"].dtype == np.dtype("int32")
        assert file["entry/scalar"][()] == 42
        np.testing.assert_array_equal(
            file["entry/values"][:], np.arange(6).reshape(2, 3)
        )
        assert file["entry/values"].attrs["units"] == "explicit units"
        np.testing.assert_array_equal(
            file["entry/values"].attrs["labels"], ["first", "second"]
        )
        assert file["entry/empty"].shape == (0, 3)
        assert file["entry/empty"].dtype == np.dtype("float32")
    assert not report.sources
    assert template.retained_datasets() == []
    assert set(output_path.parent.iterdir()) == {output_path}


def test_unmodified_template_semantically_preserves_data_metadata_properties_and_links(
    base_file, output_path
):
    before = base_file.read_bytes()
    template = mcstastox.create_template(base_file)

    report = mcstastox.write_nexus(template, output_path, read_chunk_size=5)

    with h5py.File(base_file, "r") as original, h5py.File(output_path, "r") as copied:
        names = [""]
        original.visititems(lambda name, obj: names.append(name))
        for name in names:
            left, right = original[name or "/"], copied[name or "/"]
            assert set(left.attrs) == set(right.attrs)
            for key in left.attrs:
                np.testing.assert_array_equal(left.attrs[key], right.attrs[key])
            if isinstance(left, h5py.Group):
                assert set(left) == set(right)
                continue
            assert isinstance(right, h5py.Dataset)
            assert left.shape == right.shape
            assert left.dtype == right.dtype
            np.testing.assert_array_equal(left[()], right[()])
            for key in (
                "chunks",
                "maxshape",
                "compression",
                "compression_opts",
                "shuffle",
                "fletcher32",
                "scaleoffset",
                "fillvalue",
            ):
                assert getattr(left, key) == getattr(right, key), (name, key)
        assert copied["entry/data"].id == copied["other/alias"].id
        assert copied["entry"].id == copied["entry_alias"].id
        soft = copied.get("/entry/soft", getlink=True)
        assert isinstance(soft, h5py.SoftLink)
        assert soft.path == "data"
        assert copied["entry/soft"].id == copied["entry/data"].id
    warnings = {(d.path, d.message) for d in template.validate().warnings}
    assert warnings <= {(d.path, d.message) for d in report.diagnostics}
    assert base_file.read_bytes() == before


def test_deleted_and_fully_replaced_payloads_are_never_read_even_through_aliases(
    base_file, output_path, monkeypatch
):
    before = base_file.read_bytes()
    forbidden = {
        "/entry/data",
        "/entry_alias/data",
        "/other/alias",
        "/entry/obsolete/events",
        "/entry_alias/obsolete/events",
    }
    reads, _ = _watch_io(
        monkeypatch,
        read_limits={(base_file.resolve(), path): 0 for path in forbidden},
        output_path=output_path,
    )
    template = mcstastox.create_template(base_file)
    del template["/entry/obsolete"]
    replacement = np.full((7, 3), 99, dtype="int32")
    template["/entry/data"].set_value(replacement)
    report = mcstastox.write_nexus(template, output_path)

    assert not reads
    with h5py.File(output_path, "r") as file:
        assert "obsolete" not in file["entry"]
        for path in ("entry/data", "entry_alias/data", "other/alias"):
            np.testing.assert_array_equal(file[path][:], replacement)
            assert file[path].id == file["entry/data"].id
        assert file["entry/data"].attrs["units"] == "counts"
        assert file["entry/data"].compression == "gzip"
    warnings = {(d.path, d.message) for d in template.validate().warnings}
    assert any("aliases" in message for _, message in warnings)
    assert warnings <= {(d.path, d.message) for d in report.diagnostics}
    assert base_file.read_bytes() == before


@pytest.mark.parametrize("chunked", [False, True], ids=["contiguous", "chunked"])
def test_retained_multidimensional_copies_tile_within_large_rows(
    tmp_path, output_path, monkeypatch, chunked
):
    path = tmp_path / "multidimensional.nxs"
    values = np.arange(3 * 17 * 23, dtype="int32").reshape(3, 17, 23)
    with h5py.File(path, "w") as file:
        file.create_dataset("cube", data=values, chunks=(2, 5, 7) if chunked else None)
    before = path.read_bytes()
    template = mcstastox.create_template(path)
    reads, writes = _watch_io(
        monkeypatch,
        read_limits={(path.resolve(), "/cube"): 11},
        output_path=output_path,
        write_limits={"/cube": 7},
    )

    mcstastox.write_nexus(template, output_path, read_chunk_size=11, write_chunk_size=7)

    assert sum(reads["/cube"]) == values.size
    assert sum(writes["/cube"]) == values.size
    with h5py.File(output_path, "r") as file:
        np.testing.assert_array_equal(file["cube"][:], values)
        assert file["cube"].chunks == ((2, 5, 7) if chunked else None)
    assert path.read_bytes() == before


def test_raw_source_copies_preserve_dtype_and_honor_separate_per_copy_batches(
    source_file, output_path, monkeypatch
):
    before = source_file.read_bytes()
    template = mcstastox.create_empty_template()
    target = template.add_dataset("/raw", shape=(0,), dtype="int16")
    target.copy_from("/entry2/raw", read_chunk_size=4, write_chunk_size=3, chunks=(5,))
    target.attrs["units"] = "destination units"
    template.add_dataset("/scalar").copy_from("/entry1/scalar")
    template.add_dataset("/empty").copy_from("/entry2/empty")
    template.add_dataset("/text").copy_from("/entry2/text")
    reads, writes = _watch_io(
        monkeypatch,
        read_limits={(source_file.resolve(), "/entry2/raw"): 4},
        output_path=output_path,
        write_limits={"/raw": 3},
    )

    report = mcstastox.write_nexus(
        template,
        output_path,
        source=source_file,
        read_chunk_size=100,
        write_chunk_size=100,
    )

    assert sum(reads["/entry2/raw"]) == 23
    assert sum(writes["/raw"]) == 23
    with h5py.File(output_path, "r") as file:
        np.testing.assert_array_equal(file["raw"][:], np.arange(23) + 200)
        assert file["raw"].dtype == np.dtype("int16")
        assert file["raw"].chunks == (5,)
        assert file["raw"].attrs["units"] == "destination units"
        assert file["scalar"].shape == ()
        assert file["scalar"].dtype == np.dtype("int32")
        assert file["scalar"][()] == 1
        assert file["empty"].shape == (0, 3)
        assert file["empty"].dtype == np.dtype("int16")
        assert file["text"].asstr()[:].tolist() == ["001.50", "caf\u00e9"]
    result = report.destinations["/raw"]
    assert result.shape == (23,)
    assert np.dtype(result.dtype) == np.dtype("int16")
    assert result.n_written == 23
    assert source_file.read_bytes() == before


def test_full_string_replacements_widen_without_losing_shape_or_attributes(
    base_file, output_path
):
    template = mcstastox.create_template(base_file)
    template["/entry/title"].set_value("A much longer scalar title")
    template["/entry/fixed"].set_value(["A much longer length-one string"])

    mcstastox.write_nexus(template, output_path)

    with h5py.File(output_path, "r") as file:
        assert file["entry/title"].shape == ()
        assert file["entry/title"].asstr()[()] == "A much longer scalar title"
        assert file["entry/fixed"].shape == (1,)
        assert file["entry/fixed"].asstr()[:].tolist() == [
            "A much longer length-one string"
        ]
        assert file["entry/fixed"].attrs["description"] == "retained text metadata"


def test_source_template_replaces_only_shared_detector_events_and_preserves_metadata(
    source_file, output_path, monkeypatch
):
    component_path = "/entry1/instrument/components/0000_Detector"
    event_path = f"{component_path}/output/event_data"
    aliases = ("/entry1/data/Detector", event_path, "/entry1/events")
    geometry_path = f"{component_path}/Geometry/position"
    parameter_path = f"{component_path}/parameters/xwidth"
    with h5py.File(source_file, "r+") as file:
        detector = file[event_path]
        file[aliases[0]] = detector
        detector.attrs["NX_class"] = "NXdata"
        detector.attrs["title"] = "Original detector metadata"
        file.create_dataset(geometry_path, data=[0.0, 0.0, 4.0]).attrs["units"] = "m"
        file.create_dataset(parameter_path, data=0.25).attrs["units"] = "m"
        file.create_dataset("/entry1/duration", data=12.5).attrs["units"] = "s"
    before = source_file.read_bytes()
    calls = _spy_sampler(monkeypatch)
    settings = mcstastox.SamplingSettings(31, seed=42)
    events = mcstastox.event_source(
        component_name="Detector", variables=("id", "t"), read_chunk_size=2
    ).sample(settings)
    pulse_period_s = 0.0025
    template = mcstastox.create_template(source_file)
    detector = template[aliases[0]]
    del detector["events"]
    detector.attrs["mcstas_original_variables"].set_from(
        lambda source: source.raw[event_path].attrs["variables"]
    )
    detector.attrs["NX_class"] = "NXevent_data"
    detector.attrs["variables"] = "event_id event_time_offset"
    detector.add_dataset("event_id", shape=(0,), dtype="int64").copy_from(events["id"])
    time = detector.add_dataset("event_time_offset", shape=(0,), dtype="float64")
    time.copy_from(
        events["t"]
        .map(lambda values: values % pulse_period_s, dtype="float64")
        .convert_units("s", "us")
    )
    time.attrs["units"] = "us"
    template.create_link(aliases[2], aliases[0])
    sampling = template.add_group("/entry1/sampling")
    duration = sampling.add_dataset("duration", shape=(), dtype="float64")
    duration.set_from_result(events, "effective_duration")
    duration.attrs["units"] = "s"

    report = mcstastox.write_nexus(
        template, output_path, source=source_file, write_chunk_size=3
    )

    assert len(calls) == 1
    assert calls[0][0] is settings
    sampled = calls[0][1]
    assert set(report.sources) == {events}
    result = report.sources[events]
    assert result.n_selected == result.n_written == 31
    assert result.effective_duration == pytest.approx(31 / 7)
    assert np.any(sampled["t"] >= pulse_period_s)
    preserved_paths = (
        geometry_path,
        parameter_path,
        "/entry1/duration",
        "/entry1/title",
        "/entry1/raw",
        "/entry1/instrument/components/0001_Second/output/event_data/events",
        "/entry2/instrument/components/0000_Detector/output/event_data/events",
        "/entry2/raw",
        "/entry10/instrument/components/0000_Detector/output/event_data/events",
        "/entry10/raw",
    )
    for audit in (template.retained_datasets(), report.retained_datasets):
        retained_paths = {
            path for record in audit for path in (record.path, *record.aliases)
        }
        assert retained_paths.isdisjoint(f"{alias}/events" for alias in aliases)
        assert set(preserved_paths) <= retained_paths
    with h5py.File(source_file, "r") as original, h5py.File(output_path, "r") as file:
        assert set(file) == {"entry1", "entry2", "entry10"}
        detector = file[aliases[0]]
        ids = detector["event_id"]
        time = detector["event_time_offset"]
        assert ids.shape == time.shape == (31,)
        assert ids.dtype == np.dtype("int64")
        assert time.dtype == np.dtype("float64")
        np.testing.assert_array_equal(ids[:], sampled["id"])
        np.testing.assert_array_equal(time[:], (sampled["t"] % pulse_period_s) * 1e6)
        assert time.attrs["units"] == "us"
        for alias in aliases:
            assert isinstance(file.get(alias, getlink=True), h5py.HardLink)
            group = file[alias]
            assert group.id == detector.id
            assert "events" not in group
            assert group["event_id"].id == ids.id
            assert group["event_time_offset"].id == time.id
            assert group.attrs["NX_class"] == "NXevent_data"
            assert group.attrs["variables"] == "event_id event_time_offset"
            assert group.attrs["mcstas_original_variables"] == np.bytes_("p t id w L")
            assert group.attrs["title"] == "Original detector metadata"
        for path in preserved_paths:
            assert file[path].dtype == original[path].dtype
            np.testing.assert_array_equal(file[path][()], original[path][()])
        for path in (geometry_path, parameter_path):
            assert file[path].attrs["units"] == "m"
        assert file["/entry1/simulation"].attrs["program"] == np.bytes_("3.8.6")
        assert file["/entry1/duration"].attrs["units"] == "s"
        assert file["/entry1/sampling/duration"].shape == ()
        assert file["/entry1/sampling/duration"].attrs["units"] == "s"
        assert file["/entry1/sampling/duration"][()] == pytest.approx(
            result.effective_duration
        )
    assert source_file.read_bytes() == before


def test_selected_callback_string_update_rejects_truncation_atomically(
    base_file, output_path
):
    before = base_file.read_bytes()
    template = mcstastox.create_template(base_file)
    calls = []

    def too_long(source):
        calls.append(source)
        return "too long for S3"

    template["/entry/fixed"].set_from(too_long, selection=0)

    with pytest.raises(
        (TypeError, ValueError), match=r"(?i)(truncat|string|length|fit)"
    ):
        mcstastox.write_nexus(template, output_path)

    assert len(calls) == 1
    assert not output_path.exists()
    assert list(output_path.parent.iterdir()) == []
    assert base_file.read_bytes() == before


@pytest.mark.parametrize(
    ("offset", "dtype", "message"),
    [(128, "int8", "(?i)(range|overflow)"), (0.5, "int64", "(?i)(integral|fraction)")],
    ids=["overflow", "fractional"],
)
def test_event_destination_integer_casts_fail_instead_of_wrapping_or_rounding(
    source_file, output_path, offset, dtype, message
):
    before = source_file.read_bytes()
    events = mcstastox.event_source(component_name="Detector", variables="id")
    template = mcstastox.create_empty_template()
    template.add_dataset("/ids", shape=(0,), dtype=dtype).copy_from(
        events["id"].offset(offset), dtype=dtype
    )

    with pytest.raises((ValueError, OverflowError), match=message):
        mcstastox.write_nexus(
            template, output_path, source=source_file, write_chunk_size=2
        )

    assert list(output_path.parent.iterdir()) == []
    assert source_file.read_bytes() == before


@pytest.mark.parametrize("seed", [42, None], ids=["seeded", "unseeded"])
def test_one_shared_sample_fans_out_mapped_ids_converted_time_and_duration(
    source_file, output_path, monkeypatch, seed
):
    before = source_file.read_bytes()
    calls = _spy_sampler(monkeypatch)
    settings = mcstastox.SamplingSettings(31, seed=seed)
    events = mcstastox.event_source(
        component_name="Detector", variables=("id", "t", "L"), read_chunk_size=2
    ).sample(settings)
    template = mcstastox.create_empty_template()
    template.add_group("/entry").add_group("events")
    template.add_group("/analysis")
    mapping = {0: 1001, 1: 1002, 2: 1010, 3: 1011, 4: 1020}
    batches = []

    def bounded(values):
        assert len(values) <= 3
        batches.append(len(values))
        return values

    template.add_dataset("/entry/events/id", shape=(0,), dtype="int64").copy_from(
        events["id"].map_values(mapping, dtype="int64").map(bounded, dtype="int64")
    )
    time = template.add_dataset("/entry/events/time", shape=(0,), dtype="float64")
    time.copy_from(events["t"].convert_units("s", "us"))
    time.attrs["units"] = "us"
    template.add_dataset("/analysis/wavelength", shape=(0,), dtype="float64").copy_from(
        events["L"]
    )
    duration = template.add_dataset("/entry/duration", shape=(), dtype="float64")
    duration.set_from_result(events, "effective_duration")
    duration.attrs["units"] = "s"
    template.add_dataset(
        "/analysis/duration_variance", shape=(), dtype="float64"
    ).set_from_result(events, lambda result: result.effective_duration_variance)

    report = mcstastox.write_nexus(
        template, output_path, source=source_file, write_chunk_size=3
    )

    assert len(calls) == 1
    assert calls[0][0] is settings
    sampled = calls[0][1]
    assert sum(batches) == 31
    assert set(report.sources) == {events}
    result = report.sources[events]
    assert result.n_selected == result.n_written == 31
    assert result.total_weight == 7
    assert result.total_weight_variance == 21
    assert result.effective_duration == pytest.approx(31 / 7)
    assert result.effective_duration_variance == pytest.approx((31 / 7) ** 2 * 21 / 49)
    with h5py.File(output_path, "r") as file:
        ids = file["entry/events/id"][:]
        np.testing.assert_array_equal(ids, [mapping[value] for value in sampled["id"]])
        np.testing.assert_array_equal(file["entry/events/time"][:], sampled["t"] * 1e6)
        np.testing.assert_array_equal(file["analysis/wavelength"][:], sampled["L"])
        assert len(np.unique(ids)) < len(ids)
        assert file["entry/events/time"].attrs["units"] == "us"
        assert file["entry/duration"].attrs["units"] == "s"
        assert file["entry/duration"][()] == pytest.approx(result.effective_duration)
        assert file["analysis/duration_variance"][()] == pytest.approx(
            result.effective_duration_variance
        )
    for path, dtype in (
        ("/entry/events/id", "int64"),
        ("/entry/events/time", "float64"),
        ("/analysis/wavelength", "float64"),
    ):
        destination = report.destinations[path]
        assert destination.shape == (31,)
        assert np.dtype(destination.dtype) == np.dtype(dtype)
        assert destination.n_written == 31
    assert source_file.read_bytes() == before


@pytest.mark.parametrize("ordered", [False, True])
@pytest.mark.parametrize("filter_zeros", [False, True])
def test_sampled_columns_and_table_match_direct_sampler_for_identical_read_partitions(
    source_file, output_path, ordered, filter_zeros
):
    settings = mcstastox.SamplingSettings(29, seed=17, ordered=ordered, weight_key="w")
    components = ["Second", "Empty", "Detector"]
    variables = ["id", "t", "w"] + (["p"] if filter_zeros else [])
    with mcstastox.Read(
        source_file.parent, filename=source_file.name, entry_number=2
    ) as source:
        chunks = source.file_object.iter_event_data(
            variables, component_name=components, chunk_size=2
        )
        if filter_zeros:
            chunks = (
                {key: values[chunk["p"] != 0] for key, values in chunk.items()}
                for chunk in chunks
            )
        expected = mcstastox.sample_event_chunks(chunks, settings)
    events = mcstastox.event_source(
        component_name=components,
        variables=("id", "t", "w"),
        read_chunk_size=2,
        filter_zeros=filter_zeros,
        weight_key="p",
    ).sample(settings)
    template = mcstastox.create_empty_template()
    template.add_dataset("/ids", shape=(0,), dtype="int64").copy_from(events["id"])
    template.add_dataset("/table", shape=(0, 3), dtype="float64").copy_from(
        events, columns=("t", "w", "id")
    )

    report = mcstastox.write_nexus(
        template,
        output_path,
        source=str(source_file),
        entry_number=2,
        read_chunk_size=100,
        write_chunk_size=3,
    )

    with h5py.File(output_path, "r") as file:
        np.testing.assert_array_equal(file["ids"][:], expected["id"])
        np.testing.assert_array_equal(
            file["table"][:],
            np.column_stack([expected[key] for key in ("t", "w", "id")]),
        )
        np.testing.assert_array_equal(file["table"][:, 1], np.ones(29))
        assert file["ids"].dtype == np.dtype("int64")
    result = report.sources[events]
    assert result.n_selected == result.n_written == 29
    for field in (
        "total_weight",
        "total_weight_variance",
        "effective_duration",
        "effective_duration_variance",
    ):
        assert getattr(result, field) == getattr(expected, field)
    assert report.destinations["/table"].shape == (29, 3)
    assert report.destinations["/table"].n_written == 29


@pytest.mark.parametrize("component", ["Empty", "Zero"], ids=["empty", "zero-weight"])
def test_empty_and_zero_weight_samples_keep_declared_schema_and_unavailable_statistics(
    source_file, output_path, component
):
    events = mcstastox.event_source(
        component_name=component, variables=("id", "t", "w"), read_chunk_size=2
    ).sample(mcstastox.SamplingSettings(7, seed=42, weight_key="w"))
    template = mcstastox.create_empty_template()
    template.add_dataset("/ids", shape=(0,), dtype="int64").copy_from(events["id"])
    template.add_dataset("/table", shape=(0, 3), dtype="float64").copy_from(
        events, columns=("id", "t", "w")
    )
    template.add_dataset("/duration", shape=(), dtype="float64").set_from_result(
        events,
        lambda result: (
            0.0 if result.effective_duration is None else result.effective_duration
        ),
    )

    report = mcstastox.write_nexus(template, output_path, source=source_file)

    with h5py.File(output_path, "r") as file:
        assert file["ids"].shape == (0,)
        assert file["ids"].dtype == np.dtype("int64")
        assert file["table"].shape == (0, 3)
        assert file["table"].dtype == np.dtype("float64")
        assert file["duration"][()] == 0
    result = report.sources[events]
    assert result.n_selected == result.n_written == 0
    assert result.total_weight == result.total_weight_variance == 0
    assert result.effective_duration is result.effective_duration_variance is None
    assert report.destinations["/ids"].n_written == 0
    assert report.destinations["/table"].n_written == 0


def test_unavailable_direct_duration_binding_fails_before_publication(
    source_file, output_path
):
    events = mcstastox.event_source(component_name="Empty", variables="id").sample(
        mcstastox.SamplingSettings(7, seed=42)
    )
    template = mcstastox.create_empty_template()
    template.add_dataset("/ids", shape=(0,), dtype="int64").copy_from(events["id"])
    template.add_dataset("/duration", shape=(), dtype="float64").set_from_result(
        events, "effective_duration"
    )

    with pytest.raises(
        (TypeError, ValueError), match=r"(?i)(effective_duration|unavailable|None)"
    ):
        mcstastox.write_nexus(template, output_path, source=source_file)

    assert list(output_path.parent.iterdir()) == []


@pytest.mark.parametrize("filter_zeros", [False, True])
def test_unsampled_events_stream_once_without_materializing_all_chunks(
    source_file, output_path, monkeypatch, filter_zeros
):
    events = mcstastox.event_source(
        component_name=("Second", "Empty", "Detector"),
        variables=("id", "t"),
        read_chunk_size=2,
        filter_zeros=filter_zeros,
        weight_key="w",
    )
    template = mcstastox.create_empty_template()
    transformed = 0
    iterations = []

    def bounded(values):
        nonlocal transformed
        assert len(values) <= 1
        transformed += len(values)
        return values

    def forbidden(*args, **kwargs):
        pytest.fail("Unsampled writing must not materialize the complete event stream")

    original = McStasNeXus.iter_event_data

    def observed(reader, variables, *, component_name=None, chunk_size):
        iterations.append(chunk_size)
        expected_written = 0
        for chunk in original(
            reader, variables, component_name=component_name, chunk_size=chunk_size
        ):
            expected_written += (
                np.count_nonzero(chunk["w"]) if filter_zeros else len(chunk["id"])
            )
            yield chunk
            assert transformed >= expected_written - 2, (
                "Writer collected the event stream"
            )

    monkeypatch.setattr(McStasNeXus, "iter_event_data", observed)
    monkeypatch.setattr(McStasNeXus, "get_event_data", forbidden)
    monkeypatch.setattr(mcstastox.Read, "get_event_data", forbidden)
    paths = [
        f"/entry1/instrument/components/{index:04d}_{name}/output/event_data/events"
        for index, name in ((0, "Detector"), (1, "Second"))
    ]
    reads, _ = _watch_io(
        monkeypatch,
        read_limits={(source_file.resolve(), path): 2 * 5 for path in paths},
        output_path=output_path,
    )
    template.add_dataset("/ids", shape=(0,), dtype="int64").copy_from(
        events["id"].map(bounded, dtype="int64")
    )
    template.add_dataset("/time", shape=(0,), dtype="float64").copy_from(events["t"])

    report = mcstastox.write_nexus(
        template,
        output_path,
        source=source_file,
        read_chunk_size=100,
        write_chunk_size=1,
    )

    expected = np.concatenate((_EVENTS[:, 2] + 10, _EVENTS[:, 2]))
    if filter_zeros:
        expected = expected[np.tile(_EVENTS[:, 3] != 0, 2)]
    with h5py.File(output_path, "r") as file:
        np.testing.assert_array_equal(file["ids"][:], expected)
        np.testing.assert_allclose(file["time"][:], expected * 0.001)
    assert iterations == [2]
    assert transformed == len(expected)
    assert all(sum(reads[path]) == _EVENTS.size for path in paths)
    assert report.sources[events].n_written == len(expected)
    assert report.destinations["/ids"].n_written == len(expected)


def test_row_alignment_callback_failure_cleans_partially_written_sibling_tempfile(
    source_file, output_path
):
    before = source_file.read_bytes()
    events = mcstastox.event_source(
        component_name="Detector", variables="id", read_chunk_size=3
    )
    calls = []

    def break_alignment(values):
        calls.append(len(values))
        return values if len(calls) == 1 else values[:-1]

    template = mcstastox.create_empty_template()
    template.add_dataset("/ids", shape=(0,), dtype="int64").copy_from(
        events["id"].map(break_alignment, dtype="int64")
    )

    with pytest.raises(ValueError, match=r"(?i)(row|align|preserve)"):
        mcstastox.write_nexus(
            template, output_path, source=source_file, write_chunk_size=2
        )

    assert len(calls) == 2
    assert list(output_path.parent.iterdir()) == []
    assert source_file.read_bytes() == before


def test_borrowed_read_keeps_selected_entry_and_stays_open_after_success_and_failure(
    source_file, output_path
):
    before = source_file.read_bytes()
    events = mcstastox.event_source(component_name="Detector", variables="id")
    template = mcstastox.create_empty_template()
    template.add_dataset("/ids", shape=(0,), dtype="int64").copy_from(events["id"])
    title = template.add_dataset("/title").set_from(
        lambda source: source.entry.read_text("title")
    )
    failed = output_path.with_name("failed.nxs")

    with mcstastox.Read(
        source_file.parent, filename=source_file.name, entry_number=2
    ) as borrowed:
        mcstastox.write_nexus(template, output_path, source=borrowed)
        assert borrowed.file.id.valid
        assert borrowed.file_object.entry_name == "entry2"
        with h5py.File(output_path, "r") as file:
            np.testing.assert_array_equal(file["ids"][:], _EVENTS[:, 2] + 100)
            assert file["title"].asstr()[()] == "002"

        def fail(source):
            assert source.raw is borrowed.file
            assert source.raw.id.valid
            raise RuntimeError("borrowed callback failure")

        title.set_from(fail)
        with pytest.raises(RuntimeError, match="borrowed callback failure"):
            mcstastox.write_nexus(template, failed, source=borrowed)
        assert borrowed.file.id.valid
        with pytest.raises(ValueError, match=r"(?i)(conflict|entry)"):
            mcstastox.write_nexus(template, failed, source=borrowed, entry_number=1)
        assert borrowed.file.id.valid
        assert set(output_path.parent.iterdir()) == {output_path}

    with pytest.raises(ValueError, match=r"(?i)closed"):
        mcstastox.write_nexus(template, failed, source=borrowed)
    assert set(output_path.parent.iterdir()) == {output_path}
    assert source_file.read_bytes() == before


def test_overwrite_requires_explicit_permission_and_success_replaces_the_output(
    output_path,
):
    template = mcstastox.create_empty_template()
    value = template.add_dataset("/value", value=1)
    mcstastox.write_nexus(template, output_path)
    before = output_path.read_bytes()
    value.set_value(2)

    with pytest.raises(FileExistsError):
        mcstastox.write_nexus(template, output_path)

    assert output_path.read_bytes() == before
    assert set(output_path.parent.iterdir()) == {output_path}
    mcstastox.write_nexus(template, output_path, overwrite=True)
    with h5py.File(output_path, "r") as file:
        assert file["value"][()] == 2
    assert set(output_path.parent.iterdir()) == {output_path}


def test_failed_overwrite_preserves_previous_output_and_closes_owned_source(
    source_file, output_path
):
    initial = mcstastox.create_empty_template()
    initial.add_dataset("/old", value=123)
    mcstastox.write_nexus(initial, output_path)
    before = output_path.read_bytes()
    source_before = source_file.read_bytes()
    handles = []

    def from_source(source):
        handles.append(source.raw)
        return source.entry.read("scalar")

    def reject(completed):
        assert completed.id.valid
        assert completed["new"][()] == 2
        assert Path(completed.filename).parent == output_path.parent
        assert Path(completed.filename) != output_path
        assert handles[0].id.valid
        assert output_path.read_bytes() == before
        raise RuntimeError("validator rejected completed output")

    template = mcstastox.create_empty_template()
    template.add_dataset("/new").set_from(from_source)
    with pytest.raises(RuntimeError, match="validator rejected completed output"):
        mcstastox.write_nexus(
            template,
            output_path,
            source=source_file,
            entry_number=2,
            overwrite=True,
            validators=(reject,),
        )

    assert len(handles) == 1
    assert not handles[0].id.valid
    assert output_path.read_bytes() == before
    assert source_file.read_bytes() == source_before
    assert set(output_path.parent.iterdir()) == {output_path}


@pytest.mark.parametrize("alias_kind", ["same", "symlink", "hardlink"])
def test_output_cannot_alias_either_input_even_with_overwrite_permission(
    base_file, source_file, output_path, alias_kind
):
    original = {path: path.read_bytes() for path in (base_file, source_file)}
    template = mcstastox.create_template(base_file)
    for input_path in (base_file, source_file):
        if alias_kind == "same":
            destination = input_path
        else:
            destination = output_path
            if alias_kind == "symlink":
                destination.symlink_to(input_path)
            else:
                destination.hardlink_to(input_path)
        siblings = set(destination.parent.iterdir())

        with pytest.raises(
            (ValueError, OSError), match=r"(?i)(alias|input|same|source|template)"
        ):
            mcstastox.write_nexus(
                template, destination, source=source_file, overwrite=True
            )

        assert set(destination.parent.iterdir()) == siblings
        assert destination.samefile(input_path)
        assert destination.is_symlink() == (alias_kind == "symlink")
        for path, payload in original.items():
            assert path.read_bytes() == payload
        if alias_kind != "same":
            destination.unlink()


def test_validators_see_open_completed_output_before_publication_and_receive_progress(
    source_file, output_path
):
    events = mcstastox.event_source(component_name="Detector", variables="id").sample(
        mcstastox.SamplingSettings(11, seed=42)
    )
    template = mcstastox.create_empty_template()
    template.add_dataset("/ids", shape=(0,), dtype="int64").copy_from(events["id"])
    template.add_dataset("/duration", shape=(), dtype="float64").set_from_result(
        events, "effective_duration"
    )
    template.attrs["n_selected"].set_from_result(events, "n_selected")
    validated = []
    updates = []

    def first(completed):
        assert isinstance(completed, h5py.File)
        assert completed.id.valid
        assert not output_path.exists()
        assert Path(completed.filename).parent == output_path.parent
        assert Path(completed.filename) != output_path
        assert completed["ids"].shape == (11,)
        assert completed["duration"][()] == pytest.approx(11 / 7)
        assert completed.attrs["n_selected"] == 11
        validated.append(completed)

    def second(completed):
        assert len(validated) == 1
        assert completed.id.valid
        assert not output_path.exists()
        validated.append(completed)

    mcstastox.write_nexus(
        template,
        output_path,
        source=source_file,
        validators=(first, second),
        progress=updates.append,
        write_chunk_size=3,
    )

    assert len(validated) == 2
    assert all(not handle.id.valid for handle in validated)
    assert updates
    for update in updates:
        assert isinstance(update.phase, str)
        assert update.phase
        assert update.path is None or isinstance(update.path, str)
        assert update.rows is None or (
            isinstance(update.rows, (int, np.integer)) and update.rows >= 0
        )
    assert any(update.rows is not None and update.rows > 0 for update in updates)
    assert set(output_path.parent.iterdir()) == {output_path}


def test_source_and_result_callbacks_bind_values_and_attributes_before_owned_close(
    source_file, output_path
):
    events = mcstastox.event_source(
        component_name="Detector", variables="id", read_chunk_size=2
    ).sample(mcstastox.SamplingSettings(13, seed=42, weight_key="w"))
    template = mcstastox.create_empty_template()
    handles = []
    result_calls = []

    def title(source):
        assert source.raw.id.valid
        assert source.entry_name == "entry2"
        handles.append(source.raw)
        return source.entry.read_text("title") + " rewritten"

    def duration_attribute(result):
        assert handles[0].id.valid
        result_calls.append(result.effective_duration)
        return 2 * result.effective_duration

    template.add_dataset("/title").set_from(title)
    ids = template.add_dataset("/ids", shape=(0,), dtype="int64").copy_from(
        events["id"]
    )
    ids.attrs["entry"].set_from(lambda source: source.entry_name)
    ids.attrs["double_duration"].set_from_result(events, duration_attribute)
    template.attrs["entry1_title"].set_from(
        lambda source: source.read_text("/entry1/title")
    )
    template.attrs["n_selected"].set_from_result(events, "n_selected")
    template.add_dataset("/combined", shape=(), dtype="float64").set_from_result(
        events, lambda result: result.n_selected + result.total_weight
    )

    report = mcstastox.write_nexus(
        template, output_path, source=source_file, entry_number=2
    )

    assert len(handles) == 1
    assert not handles[0].id.valid
    assert result_calls == [report.sources[events].effective_duration]
    with h5py.File(output_path, "r") as file:
        assert file["title"].asstr()[()] == "002 rewritten"
        assert file["ids"].attrs["entry"] == "entry2"
        assert file.attrs["entry1_title"] == "001"
        assert file.attrs["n_selected"] == 13
        assert file["combined"][()] == 23
        assert file["ids"].attrs["double_duration"] == pytest.approx(2 * 13 / 10)
        assert np.all(file["ids"][:] >= 100)


def test_separately_constructed_samples_are_not_deduplicated_by_equal_settings(
    source_file, output_path, monkeypatch
):
    calls = _spy_sampler(monkeypatch)
    settings = mcstastox.SamplingSettings(19, seed=42)
    original = mcstastox.event_source(component_name="Detector", variables="id")
    first, second = original.sample(settings), original.sample(settings)
    template = mcstastox.create_empty_template()
    template.add_dataset("/first", shape=(0,), dtype="int64").copy_from(first["id"])
    template.add_dataset("/second", shape=(0,), dtype="int64").copy_from(second["id"])

    report = mcstastox.write_nexus(template, output_path, source=source_file)

    assert len(calls) == 2
    assert all(call_settings is settings for call_settings, _ in calls)
    assert calls[0][1] is not calls[1][1]
    assert set(report.sources) == {first, second}
    assert report.sources[first].n_selected == report.sources[second].n_selected == 19
    assert report.sources[first].n_written == report.sources[second].n_written == 19
    with h5py.File(output_path, "r") as file:
        np.testing.assert_array_equal(file["first"][:], calls[0][1]["id"])
        np.testing.assert_array_equal(file["second"][:], calls[1][1]["id"])


def test_template_reuse_with_fixed_seed_is_independent_of_output_batch_size(
    source_file, output_path, monkeypatch
):
    calls = _spy_sampler(monkeypatch)
    events = mcstastox.event_source(
        component_name="Detector", variables=("id", "t"), read_chunk_size=2
    ).sample(mcstastox.SamplingSettings(37, seed=7))
    template = mcstastox.create_empty_template()
    batch_limit = 0
    observed_batches = []

    def bounded(values):
        assert len(values) <= batch_limit
        observed_batches.append(len(values))
        return values

    template.add_dataset("/ids", shape=(0,), dtype="int64").copy_from(events["id"])
    template.add_dataset("/time", shape=(0,), dtype="float64").copy_from(
        events["t"]
        .convert_units("s", "us")
        .scale(2)
        .offset(3)
        .map(bounded, dtype="float64")
    )
    before = template.summary()
    results = []
    for batch_limit in (1, 8):
        destination = output_path.with_name(f"batch-{batch_limit}.nxs")
        observed_batches.clear()
        report = mcstastox.write_nexus(
            template, destination, source=source_file, write_chunk_size=batch_limit
        )
        assert sum(observed_batches) == 37
        assert template.summary() == before
        with h5py.File(destination, "r") as file:
            results.append((file["ids"][:], file["time"][:], report.sources[events]))

    assert len(calls) == 2
    np.testing.assert_array_equal(results[0][0], results[1][0])
    np.testing.assert_array_equal(results[0][1], results[1][1])
    np.testing.assert_allclose(results[0][1], results[0][0] * 2000 + 3)
    for field in ("n_selected", "n_written", "total_weight", "effective_duration"):
        assert getattr(results[0][2], field) == getattr(results[1][2], field)


def test_reusing_an_unseeded_template_starts_a_fresh_sampling_execution(
    source_file, output_path, monkeypatch
):
    calls = _spy_sampler(monkeypatch)
    events = mcstastox.event_source(component_name="Detector", variables="id").sample(
        mcstastox.SamplingSettings(41, seed=None)
    )
    template = mcstastox.create_empty_template()
    template.add_dataset("/ids", shape=(0,), dtype="int64").copy_from(events["id"])
    for number in range(2):
        destination = output_path.with_name(f"unseeded-{number}.nxs")
        report = mcstastox.write_nexus(template, destination, source=source_file)
        assert len(calls) == number + 1
        with h5py.File(destination, "r") as file:
            np.testing.assert_array_equal(file["ids"][:], calls[number][1]["id"])
        assert report.sources[events].n_selected == 41
    # Fresh executions, not probabilistic assertions that random arrays must differ.
    assert calls[0][1] is not calls[1][1]


def test_unsupported_retained_references_block_publication_until_explicitly_removed(
    tmp_path, output_path, monkeypatch
):
    path = tmp_path / "references.nxs"
    with h5py.File(path, "w") as file:
        target = file.create_dataset("target", data=np.arange(4))
        file.create_dataset("refs", data=[target.ref], dtype=h5py.ref_dtype)
        file.attrs["object_reference"] = target.ref
    before = path.read_bytes()
    initial = mcstastox.create_empty_template()
    initial.add_dataset("/previous", value=123)
    mcstastox.write_nexus(initial, output_path)
    previous = output_path.read_bytes()
    template = mcstastox.create_template(path)
    reads, _ = _watch_io(
        monkeypatch,
        read_limits={(path.resolve(), "/refs"): 0},
        output_path=output_path,
    )
    with pytest.raises((TypeError, ValueError), match=r"(?i)refer") as error:
        mcstastox.write_nexus(template, output_path, overwrite=True)

    assert "/refs" in str(error.value) or "object_reference" in str(error.value)
    assert not reads
    assert output_path.read_bytes() == previous
    assert set(output_path.parent.iterdir()) == {output_path}
    del template["/refs"]
    del template.attrs["object_reference"]
    mcstastox.write_nexus(template, output_path, overwrite=True)
    with h5py.File(output_path, "r") as file:
        assert set(file) == {"target"}
        assert "object_reference" not in file.attrs
        np.testing.assert_array_equal(file["target"][:], np.arange(4))
    assert path.read_bytes() == before


def test_moved_retained_data_and_partial_updates_preserve_aliases_and_repair_soft_links(
    tmp_path, output_path, monkeypatch
):
    path = tmp_path / "move.nxs"
    values = np.arange(21, dtype="int32").reshape(7, 3)
    with h5py.File(path, "w") as file:
        data = file.create_dataset("old/data", data=values)
        data.attrs["target"] = "/old/data"
        data.attrs["unrelated"] = "/old/data"
        file.create_group("other")["alias"] = data
        file["absolute"] = h5py.SoftLink("/old/data")
        file["old/relative"] = h5py.SoftLink("data")
    before = path.read_bytes()
    template = mcstastox.create_template(path)
    template.add_group("/new")
    template.move("/old", "/new/moved", repair_links=True)
    selection = (slice(1, 6, 2), slice(0, 3, 2))
    template["/new/moved/data"].set_value(np.int32(-3), selection=selection)
    reads, _ = _watch_io(
        monkeypatch,
        read_limits={(path.resolve(), "/old/data"): 5},
        output_path=output_path,
    )

    mcstastox.write_nexus(template, output_path, read_chunk_size=5, write_chunk_size=3)

    assert reads["/old/data"]
    values[selection] = -3
    with h5py.File(output_path, "r") as file:
        assert "old" not in file
        data = file["new/moved/data"]
        np.testing.assert_array_equal(data[:], values)
        assert data.id == file["other/alias"].id
        assert data.id == file["absolute"].id == file["new/moved/relative"].id
        assert data.attrs["target"] == "/new/moved/data"
        assert data.attrs["unrelated"] == "/old/data"
        assert file.get("absolute", getlink=True).path == "/new/moved/data"
    assert path.read_bytes() == before
