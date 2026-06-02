from src.adapters.behavior.phone_hand_detector import (
    PhoneBox,
    _point_to_bbox_distance,
    _wrists_near_any_phone,
)


def test_point_inside_bbox_zero_distance():
    box = PhoneBox(10, 10, 100, 100, 0.9)
    assert _point_to_bbox_distance(50, 50, box) == 0.0


def test_point_outside_bbox_positive_distance():
    box = PhoneBox(10, 10, 100, 100, 0.9)
    d = _point_to_bbox_distance(200, 50, box)
    assert d == 100.0


def test_wrist_near_phone():
    box = PhoneBox(0, 0, 100, 100, 0.8)
    assert _wrists_near_any_phone(
        [(50, 50)],
        [box],
        distance_ratio=0.75,
        min_kpt_conf=0.3,
        kpt_conf=None,
    )


def test_wrist_far_from_phone():
    box = PhoneBox(0, 0, 50, 50, 0.8)
    assert not _wrists_near_any_phone(
        [(500, 500)],
        [box],
        distance_ratio=0.5,
        min_kpt_conf=0.3,
        kpt_conf=None,
    )
