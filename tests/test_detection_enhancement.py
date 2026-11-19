from unittest.mock import MagicMock, patch
import numpy as np
from piper_handeye.button_pose import ButtonDetection
from piper_handeye.detection_enhancement import detect_with_enhancement_fallback, enhance_detection_image


def detection(label, confidence=.9):
    return ButtonDetection(0, label, confidence, (1., 2., 20., 21.))


def test_raw_success_does_not_enhance_or_retry():
    detector = MagicMock()
    detector.detect.return_value = [detection('8')]
    image = np.zeros((32, 32, 3), np.uint8)
    with patch('piper_handeye.detection_enhancement.enhance_detection_image') as enhance:
        result, audit = detect_with_enhancement_fallback(detector, image, '8', .7)
    enhance.assert_not_called()
    assert detector.detect.call_count == 1
    assert not audit['attempted']


def test_retry_preserves_input_and_geometry():
    detector = MagicMock()
    detector.detect.side_effect = [[detection('9')], [detection('8')]]
    image = np.full((32, 64, 3), 25, np.uint8)
    original = image.copy()
    result, audit = detect_with_enhancement_fallback(detector, image, '8', .7)
    assert audit['accepted'] and audit['source'] == 'enhanced'
    assert result[0].label == '8'
    assert np.array_equal(image, original)
    assert detector.detect.call_args.args[0].shape == image.shape


def test_retry_does_not_lower_confidence_or_accept_ambiguity():
    image = np.zeros((32, 32, 3), np.uint8)
    for retry in [[detection('8', .69)], [detection('8', .9), detection('8', .89)]]:
        detector = MagicMock()
        detector.detect.side_effect = [[detection('9')], retry]
        result, audit = detect_with_enhancement_fallback(detector, image, '8', .7)
        assert result[0].label == '9'
        assert not audit['accepted']


def test_disabled_or_depth_independent_raw_failure_keeps_original():
    detector = MagicMock()
    detector.detect.return_value = []
    _, audit = detect_with_enhancement_fallback(detector, np.zeros((32,32,3),np.uint8), '8', .7, enabled=False)
    assert detector.detect.call_count == 1 and not audit['attempted']


def test_enhancement_error_keeps_raw_result():
    detector = MagicMock()
    detector.detect.return_value = [detection('9')]
    with patch('piper_handeye.detection_enhancement.enhance_detection_image', side_effect=RuntimeError('failed')):
        result, audit = detect_with_enhancement_fallback(detector, np.zeros((32,32,3),np.uint8), '8', .7)
    assert result[0].label == '9' and 'failed' in audit['error']
