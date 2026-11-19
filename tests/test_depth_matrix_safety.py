import numpy as np

from piper_handeye.depth_safety import camera_fixed_tool_ignore_mask

from piper_handeye.depth_matrix_safety import (
    build_depth_matrix_reference,
    evaluate_absolute_near_field,
    evaluate_depth_matrix_residual,
    predict_depth_matrix,
    predict_panel_plane_depth,
    downsample_depth_matrix,
)


K = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]])


def clear_stack():
    return np.full((5, 80, 100), 0.50, dtype=float)


def test_identity_prediction_matches_clear_depth_matrix():
    model = build_depth_matrix_reference(clear_stack(), K, np.eye(4), panel_depth_m=0.5)
    predicted = predict_depth_matrix(model, np.eye(4), splat_radius=0)
    result = evaluate_depth_matrix_residual(clear_stack()[0], predicted)
    assert result["matrix_coverage_fraction"] > 0.99
    assert not result["blocked_raw"]


def test_camera_motion_generates_expected_current_depth_matrix():
    model = build_depth_matrix_reference(clear_stack(), K, np.eye(4), panel_depth_m=0.5)
    current = np.eye(4)
    current[2, 3] = 0.10
    predicted = predict_depth_matrix(model, current, splat_radius=2)
    center = predicted[20:60, 25:75]
    assert np.isclose(np.nanmedian(center), 0.40, atol=1e-6)


def test_coherent_new_obstacle_trips_matrix_residual():
    model = build_depth_matrix_reference(clear_stack(), K, np.eye(4), panel_depth_m=0.5)
    predicted = predict_depth_matrix(model, np.eye(4), splat_radius=0)
    actual = clear_stack()[0]
    actual[20:35, 60:75] = 0.25
    result = evaluate_depth_matrix_residual(actual, predicted)
    assert result["matrix_largest_component_pixels"] == 225
    assert result["blocked_raw"]


def test_sparse_depth_noise_does_not_trip_matrix_residual():
    model = build_depth_matrix_reference(clear_stack(), K, np.eye(4), panel_depth_m=0.5)
    predicted = predict_depth_matrix(model, np.eye(4), splat_radius=0)
    actual = clear_stack()[0]
    actual[::4, ::4] = 0.25
    result = evaluate_depth_matrix_residual(actual, predicted)
    assert result["matrix_unexpected_pixels"] > 120
    assert result["matrix_largest_component_pixels"] == 1
    assert not result["blocked_raw"]


def test_low_prediction_coverage_is_unavailable_not_clear():
    actual = clear_stack()[0]
    predicted = np.full_like(actual, np.nan)
    result = evaluate_depth_matrix_residual(actual, predicted)
    assert not result["matrix_monitoring_available"]
    assert not result["blocked_raw"]


def test_sparse_actual_depth_is_not_misreported_as_full_coverage():
    predicted = clear_stack()[0]
    actual = np.full_like(predicted, np.nan)
    actual[::4, ::4] = 0.50
    result = evaluate_depth_matrix_residual(actual, predicted)
    assert result["matrix_actual_comparable_fraction"] == 1.0
    assert result["matrix_observed_expected_fraction"] < 0.10
    assert result["matrix_coverage_fraction"] < 0.10
    assert not result["matrix_monitoring_available"]
    assert result["reason_raw"] == "matrix_prediction_coverage_insufficient"


def test_intermittent_lower_gripper_echo_becomes_camera_rigid_layer():
    stack = clear_stack()
    stack[0, 65:75, 45:55] = 0.25
    model = build_depth_matrix_reference(stack, K, np.eye(4), panel_depth_m=0.5)
    assert np.count_nonzero(model.tool_mask) > 100
    predicted = predict_depth_matrix(model, np.eye(4), splat_radius=0)
    actual = clear_stack()[0]
    actual[65:75, 45:55] = 0.25
    result = evaluate_depth_matrix_residual(actual, predicted)
    assert not result["blocked_raw"]


def test_learned_rubber_tip_mask_is_ignored_but_nearby_obstacle_trips():
    predicted = np.full((80, 100), 0.50)
    actual = predicted.copy()
    tool_mask = np.zeros_like(actual, dtype=bool)
    tool_mask[65:75, 45:55] = True
    actual[65:75, 45:55] = 0.20
    tool = evaluate_depth_matrix_residual(
        actual, predicted, min_component_pixels=30, ignored_mask=tool_mask
    )
    assert not tool["blocked_raw"]
    assert tool["matrix_ignored_tool_pixels"] == 100

    actual[30:45, 60:75] = 0.20
    obstacle = evaluate_depth_matrix_residual(
        actual, predicted, min_component_pixels=30, ignored_mask=tool_mask
    )
    assert obstacle["blocked_raw"]
    assert obstacle["matrix_largest_component_pixels"] == 225


def test_fixed_rubber_tip_roi_is_tight_and_obstacle_outside_still_trips():
    predicted = np.full((80, 100), 0.50)
    learned = np.zeros_like(predicted, dtype=bool)
    ignored = camera_fixed_tool_ignore_mask(
        predicted.shape, learned, (0.45, 0.75, 0.55, 0.90), 2
    )
    actual = predicted.copy()
    actual[62:70, 47:53] = 0.20
    tool = evaluate_depth_matrix_residual(
        actual, predicted, min_component_pixels=30, ignored_mask=ignored
    )
    assert not tool["blocked_raw"]
    actual[25:40, 65:80] = 0.20
    obstacle = evaluate_depth_matrix_residual(
        actual, predicted, min_component_pixels=30, ignored_mask=ignored
    )
    assert obstacle["blocked_raw"]
    assert obstacle["matrix_largest_component_pixels"] == 225


def test_absolute_near_field_ignores_far_panel_and_trips_close_foam():
    depth = np.full((80, 100), 0.50)
    clear = evaluate_absolute_near_field(
        depth, reaction_max_depth_m=0.30, min_component_pixels=30
    )
    assert not clear["blocked_raw"]
    depth[25:40, 60:75] = 0.20
    blocked = evaluate_absolute_near_field(
        depth, reaction_max_depth_m=0.30, min_component_pixels=30
    )
    assert blocked["blocked_raw"]
    assert blocked["near_field_largest_component_pixels"] == 225


def test_absolute_near_field_keeps_fixed_rubber_tip_ignored():
    depth = np.full((80, 100), 0.50)
    depth[62:72, 47:53] = 0.18
    learned = np.zeros_like(depth, dtype=bool)
    ignored = camera_fixed_tool_ignore_mask(
        depth.shape, learned, (0.45, 0.75, 0.55, 0.92), 2
    )
    result = evaluate_absolute_near_field(
        depth,
        reaction_max_depth_m=0.30,
        min_component_pixels=30,
        ignored_mask=ignored,
    )
    assert not result["blocked_raw"]


def test_rgbd_edge_residual_is_ignored_but_central_obstacle_is_not():
    predicted = np.full((80, 100), 0.50)
    actual = predicted.copy()
    actual[20:40, 0:10] = 0.20
    edge = evaluate_depth_matrix_residual(
        actual, predicted, border_ignore_px=10
    )
    assert not edge["blocked_raw"]
    actual = predicted.copy()
    actual[20:35, 40:55] = 0.20
    center = evaluate_depth_matrix_residual(
        actual, predicted, border_ignore_px=10
    )
    assert center["blocked_raw"]


def test_predicted_panel_edge_band_ignores_only_edge_residual():
    predicted = np.full((80, 100), 0.60)
    predicted[:, 20:70] = 0.40
    actual = predicted.copy()
    # A reprojection strip at the known left panel-depth discontinuity.
    actual[20:45, 20:28] = 0.25
    edge = evaluate_depth_matrix_residual(
        actual, predicted, min_component_pixels=30,
        predicted_edge_ignore_dilation_px=10,
    )
    assert edge["matrix_predicted_edge_ignored_pixels"] > 0
    assert not edge["blocked_raw"]

    # A same-sized object away from every predicted discontinuity still stops.
    actual = predicted.copy()
    actual[20:45, 42:50] = 0.25
    interior = evaluate_depth_matrix_residual(
        actual, predicted, min_component_pixels=30,
        predicted_edge_ignore_dilation_px=10,
    )
    assert interior["blocked_raw"]


def test_absolute_twenty_centimeter_reaction_gate():
    predicted = np.full((80, 100), 0.50)
    near = predicted.copy()
    near[20:35, 40:55] = 0.19
    near_result = evaluate_depth_matrix_residual(
        near, predicted, reaction_max_depth_m=0.20
    )
    assert near_result["blocked_raw"]
    far = predicted.copy()
    far[20:35, 40:55] = 0.21
    far_result = evaluate_depth_matrix_residual(
        far, predicted, reaction_max_depth_m=0.20
    )
    assert far_result["matrix_residual_pixels_before_depth_gate"] == 225
    assert far_result["matrix_unexpected_pixels"] == 0
    assert not far_result["blocked_raw"]


def test_twenty_centimeter_boundary_itself_is_not_inside_gate():
    predicted = np.full((80, 100), 0.50)
    actual = predicted.copy()
    actual[20:35, 40:55] = 0.20
    result = evaluate_depth_matrix_residual(
        actual, predicted, reaction_max_depth_m=0.20
    )
    assert not result["blocked_raw"]


def test_half_resolution_near_depth_pooling_preserves_close_obstacle():
    depth = np.full((80, 100), 0.50)
    depth[20:30, 40:50] = 0.15
    reduced = downsample_depth_matrix(depth, 2, reducer="min")
    assert reduced.shape == (40, 50)
    assert np.allclose(reduced[10:15, 20:25], 0.15)


def test_reference_median_pooling_rejects_one_pixel_near_speckle():
    depth = np.full((80, 100), 0.50)
    depth[20, 40] = 0.15
    reduced = downsample_depth_matrix(depth, 2, reducer="median")
    assert np.isclose(reduced[10, 20], 0.50)


def test_small_panel_reprojection_shift_is_treated_as_known_safe_geometry():
    predicted = np.full((80, 100), 0.60)
    predicted[20:60, 30:70] = 0.28
    actual = np.full_like(predicted, 0.60)
    actual[20:60, 33:73] = 0.28
    strict = evaluate_depth_matrix_residual(
        actual, predicted, min_component_pixels=30
    )
    assert strict["blocked_raw"]
    tolerant = evaluate_depth_matrix_residual(
        actual,
        predicted,
        min_component_pixels=30,
        alignment_tolerance_px=3,
    )
    assert not tolerant["blocked_raw"]
    assert tolerant["matrix_alignment_tolerance_px"] == 3


def test_ten_pixel_disocclusion_tolerance_keeps_closer_object_visible():
    predicted = np.full((100, 140), 0.60)
    predicted[20:80, 30:90] = 0.28
    shifted = np.full_like(predicted, 0.60)
    shifted[20:80, 38:98] = 0.28
    old_tolerance = evaluate_depth_matrix_residual(
        shifted, predicted, min_component_pixels=120,
        alignment_tolerance_px=5, reaction_max_depth_m=0.30,
    )
    assert old_tolerance["blocked_raw"]
    corrected = evaluate_depth_matrix_residual(
        shifted, predicted, min_component_pixels=120,
        alignment_tolerance_px=10, reaction_max_depth_m=0.30,
    )
    assert not corrected["blocked_raw"]

    obstacle = shifted.copy()
    obstacle[40:60, 55:75] = 0.15
    guarded = evaluate_depth_matrix_residual(
        obstacle, predicted, min_component_pixels=120,
        alignment_tolerance_px=10, reaction_max_depth_m=0.30,
    )
    assert guarded["blocked_raw"]
    assert guarded["matrix_largest_component_pixels"] >= 400


def test_alignment_tolerance_still_detects_object_closer_than_panel():
    predicted = np.full((80, 100), 0.60)
    predicted[20:60, 30:70] = 0.28
    actual = predicted.copy()
    actual[30:45, 42:57] = 0.15
    result = evaluate_depth_matrix_residual(
        actual,
        predicted,
        min_component_pixels=30,
        alignment_tolerance_px=3,
        reaction_max_depth_m=0.30,
    )
    assert result["matrix_largest_component_pixels"] == 225
    assert result["blocked_raw"]


def test_alignment_tolerance_does_not_invent_support_in_disoccluded_region():
    predicted = np.full((80, 100), 0.38)
    predicted[:, 75:] = np.nan
    actual = np.full((80, 100), 0.38)
    # A fixed structure enters the newly visible region close to the old
    # prediction boundary.  There is no reference support here, so the
    # alignment neighbourhood must not fabricate a 0.38 m baseline and call
    # the structure an unexpected foreground object.
    actual[25:55, 75:85] = 0.29
    result = evaluate_depth_matrix_residual(
        actual,
        predicted,
        min_component_pixels=120,
        alignment_tolerance_px=10,
        reaction_max_depth_m=0.30,
        include_debug_arrays=True,
    )
    assert not result["blocked_raw"]
    assert result["matrix_unexpected_pixels"] == 0
    assert not np.any(
        np.isfinite(result["_debug"]["expected_depth_m"][:, 75:])
    )


def test_disocclusion_fix_still_detects_obstacle_on_supported_surface():
    predicted = np.full((80, 100), 0.38)
    predicted[:, 75:] = np.nan
    actual = np.full((80, 100), 0.38)
    actual[25:55, 45:55] = 0.15
    result = evaluate_depth_matrix_residual(
        actual,
        predicted,
        min_component_pixels=120,
        alignment_tolerance_px=10,
        reaction_max_depth_m=0.30,
    )
    assert result["blocked_raw"]
    assert result["matrix_largest_component_pixels"] == 300


def test_pose_projected_panel_plane_marks_newly_visible_panel_safe():
    model = build_depth_matrix_reference(
        clear_stack(),
        K,
        np.eye(4),
        panel_depth_m=0.5,
        panel_point_base_m=np.array([0.0, 0.0, 0.5]),
        panel_normal_base=np.array([0.0, 0.0, 1.0]),
    )
    plane = predict_panel_plane_depth(model, np.eye(4))
    assert np.allclose(plane, 0.5)
    predicted = np.full((80, 100), 0.80)
    actual = np.full_like(predicted, 0.80)
    actual[20:60, 30:70] = 0.50
    result = evaluate_depth_matrix_residual(
        actual,
        predicted,
        known_safe_plane_depth_m=plane,
        known_safe_plane_tolerance_m=0.03,
    )
    assert not result["blocked_raw"]
    assert result["matrix_known_safe_plane_pixels"] == 1600


def test_pose_projected_panel_plane_does_not_hide_closer_obstacle():
    predicted = np.full((80, 100), 0.80)
    plane = np.full_like(predicted, 0.50)
    actual = np.full_like(predicted, 0.80)
    actual[20:35, 40:55] = 0.20
    result = evaluate_depth_matrix_residual(
        actual,
        predicted,
        reaction_max_depth_m=0.30,
        known_safe_plane_depth_m=plane,
        known_safe_plane_tolerance_m=0.03,
    )
    assert result["blocked_raw"]
    assert result["matrix_largest_component_pixels"] == 225


def test_debug_arrays_are_opt_in_and_match_hazard_mask():
    predicted = np.full((80, 100), 0.50)
    actual = predicted.copy()
    actual[20:35, 40:55] = 0.20
    ordinary = evaluate_depth_matrix_residual(actual, predicted)
    assert "_debug" not in ordinary
    diagnostic = evaluate_depth_matrix_residual(
        actual, predicted, include_debug_arrays=True
    )
    debug = diagnostic["_debug"]
    assert debug["actual_depth_m"].shape == actual.shape
    assert debug["predicted_depth_m"].shape == actual.shape
    assert np.count_nonzero(debug["unexpected_mask"]) == 225


def test_absolute_near_field_debug_mask_is_opt_in():
    actual = np.full((80, 100), 0.50)
    actual[20:35, 40:55] = 0.20
    result = evaluate_absolute_near_field(
        actual,
        reaction_max_depth_m=0.30,
        min_component_pixels=120,
        include_debug_arrays=True,
    )
    assert result["blocked_raw"]
    assert np.count_nonzero(result["_debug"]["unexpected_mask"]) == 225
