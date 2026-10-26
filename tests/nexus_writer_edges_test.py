# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 McCode-dev contributors (https://github.com/mccode-dev)
"""Writer edge regressions using compact synthetic inputs, never McStas."""

import gc
import importlib
import tracemalloc
import weakref
from pathlib import Path

import h5py
import numpy as np
import pytest

import mcstastox

_EVENTS = np.array([[1, 0.001, 0], [2, 0.002, 1], [3, 0.003, 2]])


@pytest.fixture
def source_file(tmp_path):
    path = tmp_path / "source.h5"
    with h5py.File(path, "w") as file:
        entry = file.create_group("entry1")
        entry.create_group("data")
        simulation = entry.create_group("simulation")
        simulation.attrs["program"] = np.bytes_("3.8.6")
        simulation.create_group("Param")
        output = entry.create_group("instrument/components/0000_Detector/output")
        output.create_group("BINS")
        info = output.create_group("event_data")
        info.attrs["variables"] = np.bytes_("p t id")
        info.create_dataset("events", data=_EVENTS)
    return path


@pytest.fixture
def output_path(tmp_path):
    directory = tmp_path / "published"
    directory.mkdir()
    return directory / "written.nxs"


@pytest.fixture
def failed_temp_unlinks(monkeypatch, output_path):
    original = Path.unlink
    failed = []

    def fail_once(path, *args, **kwargs):
        if (
            path.parent == output_path.parent
            and path.name.startswith(f".{output_path.name}.")
            and path.suffix == ".tmp"
            and not failed
        ):
            failed.append(path)
            raise PermissionError("injected temporary cleanup failure")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_once)
    return failed


def test_external_source_dataset_cannot_overwrite_its_resolved_input(
    source_file, tmp_path
):
    dependency = tmp_path / "dependency.h5"
    with h5py.File(dependency, "w") as file:
        file.create_dataset("payload", data=np.arange(5, dtype="int32"))
    with h5py.File(source_file, "a") as file:
        file["entry1/linked"] = h5py.ExternalLink("dependency.h5", "/payload")
    before = {path: path.read_bytes() for path in (source_file, dependency)}
    siblings = set(tmp_path.iterdir())
    template = mcstastox.create_empty_template()
    template.add_dataset("/copied").copy_from("/entry1/linked")

    with pytest.raises((ValueError, OSError), match=r"(?i)(alias|input|same|source)"):
        mcstastox.write_nexus(template, dependency, source=source_file, overwrite=True)

    assert set(tmp_path.iterdir()) == siblings
    for path, payload in before.items():
        assert path.read_bytes() == payload


def test_retained_external_link_target_cannot_be_overwritten(tmp_path):
    dependency = tmp_path / "dependency.h5"
    with h5py.File(dependency, "w") as file:
        file.create_dataset("payload", data=np.arange(5))
    before = dependency.read_bytes()
    template = mcstastox.create_empty_template()
    template.create_link("/linked", h5py.ExternalLink(str(dependency), "/payload"))

    with pytest.raises(ValueError, match=r"aliases.*external input"):
        mcstastox.write_nexus(template, dependency, overwrite=True)

    assert dependency.read_bytes() == before
    assert set(tmp_path.iterdir()) == {dependency}


def test_unsampled_row_count_can_be_bound_by_result_field_name(
    source_file, output_path
):
    events = mcstastox.event_source(component_name="Detector", variables="id")
    template = mcstastox.create_empty_template()
    template.add_dataset("/count").set_from_result(events, "n_written")
    template.attrs["count"].set_from_result(events, "n_written")

    report = mcstastox.write_nexus(template, output_path, source=source_file)

    assert report.sources[events].n_written == len(_EVENTS)
    with h5py.File(output_path, "r") as file:
        assert file["count"][()] == len(_EVENTS)
        assert file.attrs["count"] == len(_EVENTS)


@pytest.mark.parametrize("patch_kind", ["literal", "callback"])
def test_selected_result_binding_runs_before_later_selected_updates(
    source_file, output_path, patch_kind
):
    events = mcstastox.event_source(component_name="Detector", variables="id")
    template = mcstastox.create_empty_template()
    target = template.add_dataset("/values", value=[0, 0])
    calls = []

    def count(result):
        calls.append("result")
        return result.n_written

    target.set_from_result(events, count, selection=0)
    if patch_kind == "literal":
        target.set_value(99, selection=0)
    else:

        def patch(source):
            calls.append("callback")
            return 99

        target.set_from(patch, selection=0)

    report = mcstastox.write_nexus(template, output_path, source=source_file)

    with h5py.File(output_path, "r") as file:
        np.testing.assert_array_equal(file["values"][:], [99, 0])
    assert report.sources[events].n_written == len(_EVENTS)
    assert calls == (["result"] if patch_kind == "literal" else ["result", "callback"])


def test_whole_result_dataset_accepts_later_selected_literal_and_callback_patches(
    source_file, output_path
):
    events = mcstastox.event_source(component_name="Detector", variables="id")
    template = mcstastox.create_empty_template()
    target = template.add_dataset("/values", shape=(0,), dtype="int64")
    target.set_from_result(events, lambda result: np.array([result.n_written, 7, 8]))
    target.set_value(99, selection=0)
    target.set_from(lambda source: 77, selection=1)

    mcstastox.write_nexus(template, output_path, source=source_file)

    with h5py.File(output_path, "r") as file:
        np.testing.assert_array_equal(file["values"][:], [99, 77, 8])


def test_whole_event_projection_accepts_ordered_selected_patches(
    source_file, output_path
):
    events = mcstastox.event_source(component_name="Detector", variables="id")
    template = mcstastox.create_empty_template()
    target = template.add_dataset("/ids", shape=(0,), dtype="int64")
    target.copy_from(events["id"])
    target.set_from_result(events, lambda result: result.n_written, selection=0)
    target.set_value(99, selection=0)
    target.set_from(lambda source: 88, selection=1)

    mcstastox.write_nexus(template, output_path, source=source_file)

    with h5py.File(output_path, "r") as file:
        np.testing.assert_array_equal(file["ids"][:], [99, 88, 2])


def test_selected_event_column_binding_remains_unsupported(source_file, output_path):
    events = mcstastox.event_source(component_name="Detector", variables="id")
    template = mcstastox.create_empty_template()
    template.add_dataset("/ids", shape=(3,), dtype="int64").copy_from(
        events["id"], selection=slice(None)
    )

    with pytest.raises((TypeError, ValueError), match=r"(?i)(event|projection|whole)"):
        mcstastox.write_nexus(template, output_path, source=source_file)

    assert list(output_path.parent.iterdir()) == []


@pytest.mark.parametrize(
    "project_events", [True, False], ids=["projected", "result-only"]
)
def test_independent_samples_release_all_arrays_before_the_next_sampler_call(
    source_file, output_path, monkeypatch, project_events
):
    sampling = importlib.import_module("mcstastox.Sampling")
    writer = importlib.import_module("mcstastox.NexusWrite")
    original = sampling.sample_event_chunks
    references = []

    def observed(chunks, settings):
        if references:
            gc.collect()
            assert all(ref() is None for ref in references[0].values()), (
                "The previous sample arrays are still alive at the next sampler call"
            )
        sampled = original(chunks, settings)
        references.append({key: weakref.ref(array) for key, array in sampled.items()})
        return sampled

    for module in (mcstastox, sampling, writer):
        if getattr(module, "sample_event_chunks", None) is original:
            monkeypatch.setattr(module, "sample_event_chunks", observed)
    original_source = mcstastox.event_source(
        component_name="Detector", variables=("p", "t", "id"), read_chunk_size=2
    )
    settings = mcstastox.SamplingSettings(11, seed=17)
    providers = (original_source.sample(settings), original_source.sample(settings))
    template = mcstastox.create_empty_template()
    for index, provider in enumerate(providers):
        template.add_dataset(f"/count{index}", shape=(), dtype="int64").set_from_result(
            provider, lambda result: result.n_written
        )
        if project_events:
            for variable in ("id", "t", "p"):
                template.add_dataset(
                    f"/sample{index}_{variable}", shape=(0,), dtype="float64"
                ).copy_from(provider[variable])

    report = mcstastox.write_nexus(
        template, output_path, source=source_file, write_chunk_size=2
    )

    assert len(references) == 2
    assert all(set(refs) == {"p", "t", "id"} for refs in references)
    gc.collect()
    assert all(ref() is None for refs in references for ref in refs.values())
    assert set(report.sources) == set(providers)
    with h5py.File(output_path, "r") as file:
        for index, provider in enumerate(providers):
            result = report.sources[provider]
            assert all(
                value is None or np.isscalar(value) for value in vars(result).values()
            )
            assert result.n_selected == result.n_written == settings.n_samples
            assert result.total_weight == 6
            assert result.total_weight_variance == 14
            assert result.effective_duration == pytest.approx(11 / 6)
            assert file[f"count{index}"].shape == ()
            assert file[f"count{index}"][()] == settings.n_samples
            if project_events:
                ids = file[f"sample{index}_id"][:]
                assert ids.shape == (settings.n_samples,)
                assert set(ids) <= {0, 1, 2}
                np.testing.assert_allclose(
                    file[f"sample{index}_t"][:], (ids + 1) * 0.001
                )
                np.testing.assert_array_equal(
                    file[f"sample{index}_p"][:], np.ones(settings.n_samples)
                )


@pytest.mark.parametrize(
    "shape", [(1_000_000,), (2,) * 64 + (1_000_000,)], ids=["flat", "long-rank"]
)
def test_tiles_first_yield_does_not_pool_large_axis_ranges(shape):
    # The private iterator is tested directly to isolate its memory invariant.
    writer = importlib.import_module("mcstastox.NexusWrite")
    gc.collect()
    tracemalloc.start()
    try:
        tiles = writer._tiles(shape, 1)
        first = next(tiles)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert first == (slice(0, 1),) * len(shape)
    assert peak < 1_000_000, f"First tile allocated {peak:,} traced bytes"
    assert next(tiles) == (slice(0, 1),) * (len(shape) - 1) + (slice(1, 2),)


@pytest.mark.parametrize(
    "nested_alias", [False, True], ids=["group", "nested-dependency"]
)
def test_acyclic_group_aliases_preserve_distinct_parents_and_nested_children(
    output_path, nested_alias
):
    template = mcstastox.create_empty_template()
    target = template.add_group("/a")
    target.attrs["role"] = "shared target"
    nested = target.add_group("nested")
    nested.attrs["depth"] = 1
    nested.add_dataset("values", value=[1, 2, 3]).attrs["units"] = "counts"
    parent = template.add_group("/z")
    parent.attrs["owner"] = "distinct parent"
    own = parent.add_group("own")
    own.attrs["role"] = "local child"
    own.add_dataset("values", value=[4, 5])
    template.create_link("/z/alias", "/a")
    if nested_alias:
        template.create_link("/z/own/second_alias", "/a/nested")

    mcstastox.write_nexus(template, output_path)

    with h5py.File(output_path, "r") as file:
        assert set(file) == {"a", "z"}
        assert file["a"].id == file["z/alias"].id
        assert file["z"].id != file["a"].id
        assert file["z"].attrs["owner"] == "distinct parent"
        assert file["z/own"].attrs["role"] == "local child"
        np.testing.assert_array_equal(file["z/own/values"][:], [4, 5])
        paths = ["a", "z/alias"]
        for path in paths:
            assert file[path].attrs["role"] == "shared target"
            assert file[f"{path}/nested"].attrs["depth"] == 1
            assert file[f"{path}/nested/values"].attrs["units"] == "counts"
            np.testing.assert_array_equal(file[f"{path}/nested/values"][:], [1, 2, 3])
        if nested_alias:
            assert file["z/own/second_alias"].id == file["a/nested"].id
            np.testing.assert_array_equal(
                file["z/own/second_alias/values"][:], [1, 2, 3]
            )


def test_cleanup_error_after_no_overwrite_publication_is_a_reported_warning(
    output_path, failed_temp_unlinks
):
    template = mcstastox.create_empty_template()
    template.add_dataset("/value", value=42)

    report = mcstastox.write_nexus(template, output_path)

    assert len(failed_temp_unlinks) == 1
    assert report.output == output_path
    assert any(
        diagnostic.severity == "warning" and "cleanup" in diagnostic.message.lower()
        for diagnostic in report.diagnostics
    )
    with h5py.File(output_path, "r") as file:
        assert set(file) == {"value"}
        assert file["value"][()] == 42


def test_cleanup_error_does_not_mask_the_original_callback_failure(
    output_path, failed_temp_unlinks
):
    original_error = RuntimeError("original callback failure")

    def fail(source):
        raise original_error

    template = mcstastox.create_empty_template()
    template.add_dataset("/value").set_from(fail)

    with pytest.raises(RuntimeError, match="original callback failure") as error:
        mcstastox.write_nexus(template, output_path)

    assert error.value is original_error
    assert len(failed_temp_unlinks) == 1
    assert not output_path.exists()
    notes = getattr(error.value, "__notes__", ())
    if notes:
        assert any("cleanup" in note.lower() for note in notes)


@pytest.mark.parametrize("binding", ["callback", "literal"])
def test_full_widening_inherited_fixed_utf8_preserves_text_encoding(
    tmp_path, output_path, binding
):
    base = tmp_path / "fixed-utf8.h5"
    with h5py.File(base, "w") as file:
        text = file.create_dataset(
            "text",
            data=np.array([b"old"], dtype=h5py.string_dtype("utf-8", length=3)),
        )
        assert h5py.check_string_dtype(text.dtype).encoding == "utf-8"
        text.attrs["description"] = "inherited text"
    before = base.read_bytes()
    template = mcstastox.create_template(base)
    replacement = np.array([b"caf\xc3\xa9"], dtype="S5")
    if binding == "callback":
        template["/text"].set_from(lambda source: replacement)
    else:
        template["/text"].set_value(replacement)

    mcstastox.write_nexus(template, output_path)

    with h5py.File(output_path, "r") as file:
        text = file["text"]
        declaration = h5py.check_string_dtype(text.dtype)
        assert declaration.encoding == "utf-8"
        assert declaration.length == 5
        assert text.shape == (1,)
        assert text.asstr()[:].tolist() == ["caf\u00e9"]
        assert text.attrs["description"] == "inherited text"
    assert base.read_bytes() == before


def test_ordinary_copy_rejects_top_level_hdf5_array_datatype_before_publication(
    source_file, output_path
):
    with h5py.File(source_file, "a") as file:
        dataset = file.create_dataset(
            "entry1/array", shape=(2,), dtype=np.dtype(("int16", (3,)))
        )
        dataset[:] = np.arange(6, dtype="int16").reshape(2, 3)
        assert dataset.dtype.subdtype is not None
    before = source_file.read_bytes()
    template = mcstastox.create_empty_template()
    template.add_dataset("/copied").copy_from("/entry1/array")

    with pytest.raises(
        (TypeError, ValueError),
        match=r"(?i)((unsupported|not supported).*(array|subdtype)|"
        r"(array|subdtype).*(unsupported|not supported))",
    ):
        mcstastox.write_nexus(template, output_path, source=source_file)

    assert list(output_path.parent.iterdir()) == []
    assert source_file.read_bytes() == before


@pytest.mark.parametrize("storage", ["virtual", "external-raw"])
def test_ordinary_copy_rejects_indirect_storage_in_preflight_without_payload_reads(
    source_file, tmp_path, output_path, monkeypatch, storage
):
    missing_payload = tmp_path / "missing-payload.h5"
    with h5py.File(source_file, "a") as file:
        if storage == "virtual":
            layout = h5py.VirtualLayout(shape=(3,), dtype="int32")
            layout[:] = h5py.VirtualSource(str(missing_payload), "/payload", shape=(3,))
            file.create_virtual_dataset("entry1/indirect", layout, fillvalue=-9)
        else:
            file.create_dataset(
                "entry1/indirect",
                shape=(3,),
                dtype="int32",
                external=[(str(missing_payload), 0, h5py.h5f.UNLIMITED)],
            )
    before = source_file.read_bytes()
    reads, callbacks = [], []
    original = h5py.Dataset.__getitem__

    def observed(dataset, selection, *args, **kwargs):
        if dataset.name == "/entry1/indirect":
            reads.append(selection)
        return original(dataset, selection, *args, **kwargs)

    def callback(source):
        callbacks.append(True)
        return 1

    monkeypatch.setattr(h5py.Dataset, "__getitem__", observed)
    template = mcstastox.create_empty_template()
    template.add_dataset("/a_callback").set_from(callback)
    template.add_dataset("/copied").copy_from("/entry1/indirect")

    with pytest.raises(
        (TypeError, ValueError), match=r"(?i)(virtual|VDS|external.*(raw|storage))"
    ):
        mcstastox.write_nexus(template, output_path, source=source_file)

    assert reads == []
    assert callbacks == []
    assert list(output_path.parent.iterdir()) == []
    assert source_file.read_bytes() == before
    assert not missing_payload.exists()
