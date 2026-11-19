#!/usr/bin/env python3
"""Offline wall-time/overlap analysis. Does not connect to any robot service."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path


LABELS = dict(capture='等待新图片', detection='YOLO检测（含前后处理及补检）',
    selection='目标选择', depth='深度与三维定位', communication='视觉RPC（含服务端等待）',
    server='视觉服务请求处理', ik='IK求解', validation='轨迹校验往返',
    planning='轨迹与按压计划生成', planning_wait='等待后台计划',
    localization='静止定位采样整体', approach='接近阶段整体', attempt='按压尝试整体',
    journal='执行日志写入', reference='按前OFF参考采集', light='灯态图片计算与进程往返',
    motion='执行调用整体（含监督及回执）', motion_window='接近执行窗口（含可能的安全暂停）', idle='视觉空闲节流等待')


def merged(intervals):
    result = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if result and start <= result[-1][1]:
            result[-1][1] = max(result[-1][1], end)
        else:
            result.append([start, end])
    return result


def union_seconds(intervals):
    return sum(end-start for start, end in merged(intervals))


def overlap_seconds(first, second):
    a, b = merged(first), merged(second)
    i = j = 0
    total = 0.
    while i < len(a) and j < len(b):
        total += max(0., min(a[i][1], b[j][1])-max(a[i][0], b[j][0]))
        if a[i][1] <= b[j][1]:
            i += 1
        else:
            j += 1
    return total


def analyze(rows):
    sessions = [r for r in rows if r.get('kind') == 'session']
    if not sessions:
        raise ValueError('missing task session timing; cannot establish a comparable clock domain')
    domain = sessions[0]['clock_domain']
    local = [r for r in sessions if r.get('clock_domain') == domain]
    start = min(r['start_monotonic_sec'] for r in local)
    end = max(r['end_monotonic_sec'] for r in local)
    categories, names = defaultdict(list), defaultdict(list)
    valid, foreign, malformed = [], 0, 0
    for row in rows:
        if row.get('kind') != 'span':
            continue
        if row.get('clock_domain') != domain:
            foreign += 1
            continue
        try:
            a, b = float(row['start_monotonic_sec']), float(row['end_monotonic_sec'])
            if not math.isfinite(a+b) or b < a:
                raise ValueError('invalid interval')
            a, b = max(start, a), min(end, b)
            if b < a:
                continue
            categories[row['category']].append((a, b))
            names[row['name']].append((a, b))
            valid.append(dict(row, clipped_start=a, clipped_end=b))
        except (KeyError, TypeError, ValueError):
            malformed += 1
    summary = dict(clock_domain=domain, task_wall_sec=end-start,
        start_monotonic_sec=start, end_monotonic_sec=end,
        comparable_spans=len(valid), foreign_clock_spans_excluded=foreign,
        malformed_spans_excluded=malformed,
        dropped_spans=sum(r.get('dropped_spans', 0) for r in sessions),
        diagnostic_errors=sum(r.get('diagnostic_errors', 0) for r in sessions),
        categories={k:dict(label=LABELS.get(k, k), count=len(v),
            union_sec=union_seconds(v)) for k, v in sorted(categories.items())},
        stages={k:dict(count=len(v), inclusive_sum_sec=sum(b-a for a,b in v),
            union_sec=union_seconds(v), max_sec=max(b-a for a,b in v),
            mean_sec=sum(b-a for a,b in v)/len(v)) for k,v in sorted(names.items())},
        overlap_sec={a+' / '+b:overlap_seconds(categories[a],categories[b])
            for i,a in enumerate(sorted(categories)) for b in sorted(categories)[i+1:]
            if overlap_seconds(categories[a],categories[b]) > 0},
        limitations=[
            '各类别存在包含及并行关系，不能相加当作总耗时；重叠时间不等于可节省时间。',
            '通信往返含服务处理、序列化和调度，不是纯网络延迟。',
            '轨迹校验为客户端观察的往返，未拆分OrangePi内部MoveIt计算。',
            'YOLO为主机观察的检测调用耗时，不是独立GPU内核时间。',
            '只包含被任务读取的视觉样本；不能据此统计所有相机帧或源服务总体帧率。',
            '未插桩的空白时间不能自动归因通信或处理器空闲。'])
    return summary, valid


def write_analysis(trace, output):
    rows = [json.loads(line) for line in Path(trace).read_text().splitlines() if line.strip()]
    summary, spans = analyze(rows)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output/'timing_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    lines = ['定位与按压计时分析', f"任务进程观察范围：{summary['task_wall_sec']:.3f}秒", '',
             '各类别覆盖时间（重叠部分不可重复相加）：']
    for value in summary['categories'].values():
        lines.append(f"{value['label']}：{value['union_sec']:.4f}秒，{value['count']}条")
    lines += ['', '时间重叠：']
    lines += [f'{k}：{v:.4f}秒' for k,v in summary['overlap_sec'].items()]
    lines += ['', '解释边界：', *summary['limitations']]
    (output/'耗时分析.txt').write_text('\n'.join(lines)+'\n')
    events = []
    for row in spans:
        events.append(dict(name=row['name'], cat=row['category'], ph='X',
            pid=row.get('source_pid', row.get('pid', 0)),
            tid=row.get('source_thread_id', row.get('thread_id', 0)),
            ts=(row['clipped_start']-summary['start_monotonic_sec'])*1e6,
            dur=(row['clipped_end']-row['clipped_start'])*1e6,
            args={k:row[k] for k in ['status','frame_sequence','source','scope'] if k in row}))
    (output/'timing_trace.json').write_text(json.dumps(dict(traceEvents=events, displayTimeUnit='ms'), ensure_ascii=False))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trace', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = write_analysis(args.trace, args.output)
    print(json.dumps(dict(output=str(args.output.resolve()), task_wall_sec=result['task_wall_sec']), ensure_ascii=False))


if __name__ == '__main__':
    main()
