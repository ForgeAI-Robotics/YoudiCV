#!/usr/bin/env bash
set -u

project_dir=/home/qianjue/contral_test
python_bin=/home/qianjue/anaconda3/envs/piper-handeye/bin/python
button_label=${1:?button label is required}
stamp=$(date +%Y%m%d_%H%M%S)
output_dir="$project_dir/whole_sequence_videos"
video_path="$output_dir/button${button_label}_${stamp}.mp4"
log_path="$output_dir/button${button_label}_${stamp}.log"
manifest_path="$output_dir/latest_button${button_label}_recorded_test.txt"
recorder_pid=

mkdir -p "$output_dir"
printf 'video=%s\nlog=%s\n' "$video_path" "$log_path" > "$manifest_path"

cleanup_recorder() {
    if [[ -n "${recorder_pid}" ]] && kill -0 "$recorder_pid" 2>/dev/null; then
        kill -TERM "$recorder_pid"
        wait "$recorder_pid"
    fi
    recorder_pid=
}

terminate() {
    cleanup_recorder
    exit 143
}

trap cleanup_recorder EXIT
trap terminate TERM INT

PYTHONPATH="$project_dir" "$python_bin" \
    "$project_dir/tools/record_vision_rpc_video.py" \
    --config "$project_dir/piper_button_pose.yaml" \
    --output "$video_path" \
    --target-label "$button_label" \
    --fps 5 >> "$log_path" 2>&1 &
recorder_pid=$!

# Confirm that the recorder has received and opened its first RGB frame before
# allowing physical motion to begin.
partial_path="${video_path%.mp4}.partial.mp4"
for _ in $(seq 1 50); do
    [[ -f "$partial_path" ]] && break
    kill -0 "$recorder_pid" 2>/dev/null || exit 70
    sleep 0.1
done
[[ -f "$partial_path" ]] || exit 71

"$python_bin" "$project_dir/run_button_light_closed_loop.py" "$button_label" \
    --execute \
    --acknowledge-clear-workspace \
    --pause-depth-near-button \
    --depth-degraded-entry-distance-m 0.15 \
    --max-press-attempts 3 >> "$log_path" 2>&1
test_status=$?

cleanup_recorder
exit "$test_status"
