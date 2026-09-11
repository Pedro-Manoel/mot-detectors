import math

from core.metrics import block, reconstruct_total, run_timing_fields


def test_block_avg_std_fps():
    avg, std, fps = block([2.0, 4.0])
    assert avg == 3.0
    assert math.isclose(std, math.sqrt(2.0))
    assert math.isclose(fps, 1000.0 / 3.0)


def test_block_single_value_has_no_std():
    avg, std, fps = block([5.0])
    assert avg == 5.0
    assert std is None
    assert math.isclose(fps, 200.0)


def test_block_empty_is_none():
    assert block([]) == (None, None, None)


def test_reconstruct_total_is_per_frame_sum():
    assert reconstruct_total([1.0, 1.0], [2.0, 4.0], [1.0, 1.0]) == [4.0, 6.0]


def test_run_timing_fields_reconstructs_total_and_components():
    f = run_timing_fields(
        imread=[1.0, 1.0], det=[2.0, 4.0], track=[1.0, 1.0],
        pre=[0.5, 0.5], inf=[1.5, 3.5], post=[0.1, 0.1],
    )
    assert f["avg_det_ms"] == 3.0
    assert math.isclose(f["fps_det"], 1000.0 / 3.0)
    assert f["avg_track_ms"] == 1.0
    assert math.isclose(f["fps_track"], 1000.0)
    assert f["avg_total_ms"] == 5.0
    assert math.isclose(f["fps_total"], 200.0)
    assert math.isclose(f["std_total_ms"], math.sqrt(2.0))
    assert f["avg_inference_ms"] == 2.5
    assert f["avg_preprocess_ms"] == 0.5
    assert f["frame1_det_ms"] == 2.0
    assert f["frame1_total_ms"] == 4.0
    assert f["frame1_postprocess_ms"] == 0.1
