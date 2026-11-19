import pytest

from button_light_evaluation.evaluate import box_iou, load_manifest, summarize


def test_manifest_resolves_images_and_optional_bbox(tmp_path):
    image = tmp_path / "one.jpg"
    image.write_bytes(b"image")
    manifest = tmp_path / "data.csv"
    manifest.write_text(
        "image,target,state,gt_x1,gt_y1,gt_x2,gt_y2\n"
        "one.jpg,10,on,1,2,11,12\n", encoding="utf-8"
    )
    sample = load_manifest(manifest)[0]
    assert sample.image == image
    assert sample.state == "ON"
    assert sample.gt_bbox == (1.0, 2.0, 11.0, 12.0)


def test_box_iou():
    assert box_iou((0, 0, 10, 10), (5, 5, 15, 15)) == pytest.approx(25 / 175)


def test_summary_separates_detection_and_state_metrics():
    rows = [
        {"truth_state": "ON", "predicted_state": "ON", "target_detected": True,
         "gt_bbox": [0, 0, 10, 10], "bbox_iou": 0.8},
        {"truth_state": "OFF", "predicted_state": "UNCERTAIN", "target_detected": True,
         "gt_bbox": None},
        {"truth_state": "ON", "predicted_state": "DETECTION_MISS", "target_detected": False,
         "gt_bbox": [0, 0, 10, 10]},
    ]
    result = summarize(rows, 1.5, 0.5)
    assert result["exact_state_accuracy"] == pytest.approx(1 / 3)
    assert result["exact_state_accuracy_wilson_95"][0] < 1 / 3
    assert result["exact_state_accuracy_wilson_95"][1] > 1 / 3
    assert result["decisive_coverage"] == pytest.approx(1 / 3)
    assert result["decisive_accuracy"] == 1.0
    assert result["target_detection_rate"] == pytest.approx(2 / 3)
    assert result["bbox_recall_at_iou"] == pytest.approx(0.5)
    assert result["per_class"]["ON"]["recall"] == pytest.approx(0.5)
    assert result["per_class"]["OFF"]["recall"] == 0.0
