"""Draw saved inference outputs without running another detector."""
import cv2


def draw_recognition(image, detections, *, caption, target_label):
    # Enlarge only the review artifact, not the detector input or coordinates.
    canvas = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
    for detection in sorted(detections, key=lambda d: float(d['confidence'])):
        confidence = float(detection['confidence'])
        x1, y1, x2, y2 = [round(float(v) * 2) for v in detection['bbox_xyxy']]
        color = (0, 220, 0) if confidence >= .70 else (0, 170, 255)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cv2.putText(canvas, f"{detection['label']} {confidence:.2f}",
                    (x1, max(18, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                    .55, color, 1, cv2.LINE_AA)
    cv2.putText(canvas, f"{caption} target={target_label} boxes={len(detections)}",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 255), 2, cv2.LINE_AA)
    return canvas
