"""Conservative UP/DOWN arrow evidence for local-only highlight conflicts.

This does not classify all lamp states or segment arbitrary symbols. It can
exclude a thin vertical highlight only when two visible arrowhead diagonals
provide independent, non-luminous evidence. Unsupported/ambiguous geometry
remains uncertain. All measurements use original pixels, not enlarged images.
"""
import cv2
import numpy as np


def _geometry(gray, target, ring_median):
    height, width = gray.shape
    yy, xx = np.mgrid[:height, :width]
    xx = (xx + .5) / width
    yy = (yy + .5) / height
    if target == 'up':
        yy = 1 - yy
    best = None
    for dx in (-.04, 0., .04):
        for dy in (-.04, 0., .04):
            for scale in (.85, 1., 1.15):
                x = (xx - .5 - dx) / scale + .5
                y = (yy - .5 - dy) / scale + .5
                distance = (y - .67 + np.abs(x - .5)) / np.sqrt(2)
                wings = []
                for side in (-1, 1):
                    span = (side * (x - .5) > .07) & (side * (x - .5) < .21)
                    core = span & (np.abs(distance) < .035)
                    background = span & (np.abs(distance) > .065) & (np.abs(distance) < .11)
                    if core.sum() < 5 or background.sum() < 5:
                        break
                    median = float(np.median(gray[core]))
                    base = float(np.median(gray[background]))
                    wings.append(dict(pixels=int(core.sum()), median_gray=median,
                        face_contrast=median - base, ring_contrast=median - ring_median,
                        bright_fraction=float(np.mean(gray[core] > ring_median + 30.))))
                if len(wings) != 2:
                    continue
                # Locate the two diagonals by their shape contrast, not by
                # whichever candidate would produce the darkest OFF result.
                score = min(wing['face_contrast'] for wing in wings)
                if best is None or score > best[0]['shape_contrast']:
                    arrow = ((np.abs(x - .5) < .045) & (y > .23) & (y < .64)) | (
                        (np.abs(distance) < .045) & (np.abs(x - .5) < .23))
                    best = (dict(shape_contrast=score, dx=dx, dy=dy, scale=scale,
                                 wings=wings), arrow)
    return best


def resolve_arrow_highlight(gray, highlight_mask, target, ring_median, legacy_state):
    """Resolve only a local ON conflicting with the existing overall result.

    OFF requires ALL of: valid original resolution/geometry, both visible
    arrowhead diagonals, only narrow vertical ON candidates, >=65% of their
    pixels outside the fitted arrow, and BOTH diagonals below the conservative
    emission checks. No rejected/unknown sample is promoted to OFF.
    """
    audit = dict(policy='arrow_diagonals_v1', applied=True, target=target,
                 raw_local_state='ON', legacy_state=legacy_state,
                 min_original_size_px=28, min_shape_contrast=20.,
                 max_off_wing_ring_contrast=20., max_off_wing_bright_fraction=.5,
                 min_highlight_outside_arrow_fraction=.65)
    height, width = gray.shape
    if (min(height, width) < 1 or highlight_mask.shape != gray.shape
            or not np.any(highlight_mask)):
        return 'UNCERTAIN', {**audit, 'reason': 'arrow_geometry_unavailable'}
    # A glare stripe can be split by a few dark pixels. Closing is used only
    # to group aligned fragments; all area/brightness/outside measurements
    # below still use ORIGINAL highlighted pixels, never the filled gaps.
    kernel_height = max(3, int(np.ceil(height * .10)))
    kernel_height += 1 - kernel_height % 2
    grouped = cv2.morphologyEx(highlight_mask.astype(np.uint8), cv2.MORPH_CLOSE,
        np.ones((kernel_height, 1), np.uint8))
    grouped |= highlight_mask.astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        grouped, connectivity=8)
    stripes = []
    component_masks = []
    for label in range(1, count):
        pixels = (labels == label) & highlight_mask
        ys, xs = np.nonzero(pixels)
        if not len(xs):
            continue
        x, y = int(xs.min()), int(ys.min())
        w, h = int(xs.max()) - x + 1, int(ys.max()) - y + 1
        area = int(pixels.sum())
        thin = w / width <= .08 and h / height >= .16 and h / w >= 3.
        stripes.append(dict(bbox_xyxy=[x, y, x+w, y+h], area_px=area, thin_vertical=thin))
        component_masks.append(pixels)
    audit['highlights'] = stripes
    if not all(stripe['thin_vertical'] for stripe in stripes):
        # This is a narrowly scoped rejection of decorative vertical stripes,
        # not a replacement for the existing general local-luminance policy.
        return 'ON', {**audit, 'applied': False, 'reason': 'non_stripe_candidate_unchanged'}
    if min(height, width) < 28 or not .75 <= width / height <= 1.35:
        return 'UNCERTAIN', {**audit, 'reason': 'arrow_geometry_unavailable'}
    fitted = _geometry(gray, target, ring_median)
    if fitted is None or fitted[0]['shape_contrast'] < 20.:
        return 'UNCERTAIN', {**audit, 'reason': 'arrow_diagonals_not_resolved',
                             'geometry': fitted[0] if fitted is not None else None}
    geometry, arrow = fitted
    audit['geometry'] = geometry
    if all(wing['ring_contrast'] >= 30. and wing['bright_fraction'] >= .5
           for wing in geometry['wings']):
        return 'ON', {**audit, 'reason': 'both_arrow_diagonals_bright'}
    for stripe, pixels in zip(stripes, component_masks):
        stripe['outside_arrow_fraction'] = float(np.mean(~arrow[pixels]))
    isolated_stripes = bool(stripes) and all(
        stripe['thin_vertical'] and stripe['outside_arrow_fraction'] >= .65 for stripe in stripes)
    both_wings_not_emitting = all(wing['ring_contrast'] <= 20. and wing['bright_fraction'] <= .5
                                  for wing in geometry['wings'])
    if isolated_stripes and both_wings_not_emitting and legacy_state == 'OFF':
        return 'OFF', {**audit, 'reason': 'isolated_stripe_outside_unlit_arrow'}
    return 'UNCERTAIN', {**audit, 'reason': 'arrow_highlight_not_resolved'}
