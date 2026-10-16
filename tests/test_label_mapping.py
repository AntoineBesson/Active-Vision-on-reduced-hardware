"""Unit tests for the dataset-agnostic, config-driven label mapping."""

from __future__ import annotations

import json

import numpy as np
import pytest

from data.label_mapping import LabelMappingConfig, SegmentsIndex, SemanticLabelMapper, decode_rgb_ids

CLASSES = ["a", "b", "c", "d"]


def _rgb(ids: np.ndarray) -> np.ndarray:
    ids = np.asarray(ids, dtype=np.int64)
    return np.stack([ids % 256, (ids // 256) % 256, ids // 65536], axis=-1).astype(np.uint8)


def test_decode_rgb_ids_formula():
    ids = np.array([[0, 5, 256, 65536 * 3 + 7]])
    np.testing.assert_array_equal(decode_rgb_ids(_rgb(ids)), ids)
    with pytest.raises(ValueError):
        decode_rgb_ids(np.zeros((2, 2), np.uint8))


def test_identity_rule_out_of_range_is_ignored():
    m = SemanticLabelMapper(LabelMappingConfig(class_names=CLASSES, encoding="gray", id_rule="identity"))
    raw = np.array([[0, 1, 3, 4, 255]], dtype=np.uint8)
    out = m(raw)
    assert out.dtype == np.uint8
    np.testing.assert_array_equal(out, [[0, 1, 3, 255, 255]])


def test_divisor_rule_stuff_things_void():
    m = SemanticLabelMapper(LabelMappingConfig(class_names=CLASSES, encoding="gray", id_rule="divisor"))
    # stuff id 1, things 2*1000+0 (object id 0 is legal) and 3*1000+7, void 255.
    raw = np.array([[1, 2000, 3007, 255]], dtype=np.uint16)
    np.testing.assert_array_equal(m(raw), [[1, 2, 3, 255]])


def test_category_to_train_id_table_for_another_dataset():
    # e.g. a dataset whose raw class ids are 7, 8, 26 (unlisted ids are void).
    cfg = LabelMappingConfig(
        class_names=["road", "sidewalk", "car"],
        encoding="gray",
        id_rule="divisor",
        category_to_train_id={7: 0, 8: 1, 26: 2},
    )
    raw = np.array([[7, 8, 26001, 26, 9, 0]], dtype=np.int32)
    np.testing.assert_array_equal(SemanticLabelMapper(cfg)(raw), [[0, 1, 2, 2, 255, 255]])


def test_category_to_train_id_rejects_out_of_range_train_ids():
    with pytest.raises(ValueError, match="outside"):
        LabelMappingConfig(class_names=CLASSES, category_to_train_id={1: 7})
    # mapping explicitly to ignore is allowed
    LabelMappingConfig(class_names=CLASSES, category_to_train_id={1: 255})


def _write_segments(tmp_path, segments):
    path = tmp_path / "pan.json"
    path.write_text(
        json.dumps(
            {
                "annotations": [
                    {"image_id": "img0", "file_name": "img0_final_mask.png", "segments_info": segments}
                ],
                "categories": [],
            }
        )
    )
    return path


def test_segments_json_rule(tmp_path):
    big = 8405120  # a 24-bit RGB colour id (forces the np.unique path)
    path = _write_segments(
        tmp_path,
        [{"id": big, "category_id": 0}, {"id": 4605510, "category_id": 2}, {"id": 3937500, "category_id": 3}],
    )
    cfg = LabelMappingConfig(class_names=CLASSES, encoding="rgb", id_rule="segments_json", segments_json=str(path))
    m = SemanticLabelMapper(cfg)
    ids = np.array([[big, 4605510, 3937500, 0, 123]])  # 0 = COCO VOID, 123 = not in JSON
    out = m(_rgb(ids), annotation_name="img0_final_mask.png")
    np.testing.assert_array_equal(out, [[0, 2, 3, 255, 255]])
    # Lookup also works by stem and by image id.
    np.testing.assert_array_equal(m(_rgb(ids), annotation_name="img0_final_mask"), out)
    np.testing.assert_array_equal(m(_rgb(ids), annotation_name="img0"), out)
    with pytest.raises(KeyError, match="no entry"):
        m(_rgb(ids), annotation_name="other.png")
    with pytest.raises(ValueError, match="file name"):
        m(_rgb(ids))


def test_segments_json_small_ids_use_dense_path(tmp_path):
    path = _write_segments(tmp_path, [{"id": 1, "category_id": 1}, {"id": 2, "category_id": 3}])
    cfg = LabelMappingConfig(class_names=CLASSES, encoding="rgb", id_rule="segments_json", segments_json=str(path))
    out = SemanticLabelMapper(cfg)(_rgb(np.array([[1, 2, 0]])), annotation_name="img0_final_mask.png")
    np.testing.assert_array_equal(out, [[1, 3, 255]])


def test_missing_segments_json_is_a_clear_error(tmp_path):
    cfg = LabelMappingConfig(
        class_names=CLASSES, encoding="rgb", id_rule="segments_json", segments_json=str(tmp_path / "nope.json")
    )
    with pytest.raises(FileNotFoundError, match="segments JSON not found"):
        SemanticLabelMapper(cfg)(np.zeros((1, 1, 3), np.uint8), annotation_name="x.png")
    with pytest.raises(ValueError, match="requires segments_json"):
        LabelMappingConfig(class_names=CLASSES, id_rule="segments_json")


def test_dense_and_unique_paths_agree():
    m = SemanticLabelMapper(LabelMappingConfig(class_names=CLASSES, encoding="gray", id_rule="divisor"))
    rng = np.random.default_rng(0)
    small = rng.choice([0, 1, 2000, 3001, 255], size=(6, 7))
    big = small.copy()
    big[0, 0] = 70000  # pushes the max over the dense-LUT limit -> unique path
    np.testing.assert_array_equal(m(small)[1:], m(big)[1:])
    assert m(big)[0, 0] == 255  # 70000 // 1000 = 70 is not a class


def test_void_ids_and_auto_encoding():
    m = SemanticLabelMapper(LabelMappingConfig(class_names=CLASSES, id_rule="identity", void_ids=[2]))
    np.testing.assert_array_equal(m(np.array([[1, 2, 3]], np.uint8)), [[1, 255, 3]])  # auto -> gray
    assert m(_rgb(np.array([[1, 2, 3]]))).tolist() == [[1, 255, 3]]  # auto -> rgb


def test_explain_rows():
    m = SemanticLabelMapper(LabelMappingConfig(class_names=CLASSES, encoding="gray", id_rule="divisor"))
    rows = m.explain([3001, 1, 255, 1])
    assert rows == [
        {"id": 1, "category": 1, "train_id": 1, "class": "b"},
        {"id": 255, "category": 255, "train_id": 255, "class": "<ignore>"},
        {"id": 3001, "category": 3, "train_id": 3, "class": "d"},
    ]


def test_config_validation_and_round_trip():
    with pytest.raises(ValueError, match="encoding"):
        LabelMappingConfig(class_names=CLASSES, encoding="png")
    with pytest.raises(ValueError, match="id_rule"):
        LabelMappingConfig(class_names=CLASSES, id_rule="guess")
    with pytest.raises(ValueError, match="label_divisor"):
        LabelMappingConfig(class_names=CLASSES, label_divisor=0)
    with pytest.raises(ValueError, match="unknown label_mapping keys"):
        LabelMappingConfig.from_dict({"class_names": CLASSES, "colours": []})
    with pytest.raises(ValueError, match="at least one class"):
        LabelMappingConfig(class_names=[])
    cfg = LabelMappingConfig(class_names=CLASSES, category_to_train_id={"5": "1"})  # e.g. from JSON
    assert cfg.category_to_train_id == {5: 1}
    assert LabelMappingConfig.from_dict(cfg.to_dict()) == cfg


def test_wider_label_dtype_when_ignore_does_not_fit_uint8():
    cfg = LabelMappingConfig(class_names=CLASSES, encoding="gray", ignore_index=-1)
    out = SemanticLabelMapper(cfg)(np.array([[0, 9]], np.uint8))
    assert out.dtype == np.int16
    np.testing.assert_array_equal(out, [[0, -1]])


def test_segments_index_len(tmp_path):
    path = _write_segments(tmp_path, [{"id": 1, "category_id": 1}])
    idx = SegmentsIndex(path)
    assert len(idx) == 1
    assert idx.lookup("img0_final_mask.png") == {1: 1}
