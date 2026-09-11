import numpy as np
from core.detect import parse_ultralytics_results, PERSON_CLASS_INDEX
from core.track import parse_track_outputs


class _T:
    def __init__(self, v):
        self.v = np.asarray(v, dtype=np.float32)
    def cpu(self):
        return self
    def numpy(self):
        return self.v


class _Boxes:
    def __init__(self, xyxy, conf, cls):
        self.xyxy = _T(np.asarray(xyxy, dtype=np.float32).reshape(-1, 4))
        self.conf = _T(conf)
        self.cls = _T(cls)
    def __len__(self):
        return len(self.conf.v)


class _Result:
    def __init__(self, boxes):
        self.boxes = boxes


def test_parse_filters_class_and_conf():
    boxes = _Boxes(
        xyxy=[[0, 0, 10, 20], [5, 5, 15, 25], [1, 1, 9, 9]],
        conf=[0.9, 0.10, 0.8],
        cls=[0, 0, 1],
    )
    dets = parse_ultralytics_results([_Result(boxes)], conf_threshold=0.25)
    assert dets.shape == (1, 6)
    assert dets[0, 4] == np.float32(0.9)
    assert dets[0, 5] == PERSON_CLASS_INDEX


def test_parse_empty():
    dets = parse_ultralytics_results([_Result(_Boxes([], [], []))], 0.25)
    assert dets.shape == (0, 6)


def test_track_outputs_to_mot_rows():
    outputs = np.array([[10, 20, 30, 60, 7, 0.88, 0, 0]], dtype=np.float32)
    rows = parse_track_outputs(frame_id=3, outputs=outputs)
    assert len(rows) == 1
    assert rows[0][:2] == [3, 7]
    assert [round(x, 2) for x in rows[0][2:7]] == [10.0, 20.0, 20.0, 40.0, 0.88]
    assert rows[0][7:] == [-1, -1, -1]


def test_track_outputs_drops_nonpositive_wh():
    outputs = np.array([[10, 20, 10, 20, 1, 0.5, 0, 0]], dtype=np.float32)
    assert parse_track_outputs(1, outputs) == []
