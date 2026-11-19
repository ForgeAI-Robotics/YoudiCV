#!/usr/bin/env python3
"""One authorized first-leg navigation test using the vendor Android protocol.

Wait for a fresh Android floor command, then wait ten monotonic seconds before
sending exactly one named navigation goal. Never starts the arm or other legs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import socket
import sys
import time
import xmlrpc.client

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from piper_handeye.android_agx_protocol import encode_packet, decode_packet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--goal', default='liftout')
    parser.add_argument('--journal', type=Path, required=True)
    args = parser.parse_args()
    if args.goal != 'liftout':
        parser.error('本次仅允许导航至已确认的外呼命名点 liftout')
    master = xmlrpc.client.ServerProxy('http://192.168.168.100:11311')
    code, _, current_map = master.getParam('/agx_first_leg_test', '/multiple_map_switching_node/UD_map_yaml')
    if code != 1 or current_map != 'jxb_105.yaml':
        raise RuntimeError(f'当前地图与本次已查询点位不一致：{current_map}')
    args.journal.parent.mkdir(parents=True, exist_ok=True)
    with args.journal.open('a', encoding='utf-8', buffering=1) as out:
        def record(event, **data):
            item = dict(event=event, unix_sec=time.time(), monotonic_sec=time.monotonic(), **data)
            line = json.dumps(item, ensure_ascii=False, separators=(',', ':'))
            out.write(line+'\n'); print(line, flush=True)

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(('192.168.168.162', 5005)); sock.settimeout(.2)
        record('ready', execute=args.execute, goal=args.goal, map=current_map,
               message='请在Android面板选楼层并运行；收到后等待10秒')
        target = None; deadline = None; sent_at = None; finished = False
        sequence = int(time.time())
        end = time.monotonic()+120
        try:
            while time.monotonic() < end:
                if deadline is not None and sent_at is None and time.monotonic() >= deadline:
                    wire = encode_packet(sequence, 'active_result', {'command':1, 'value':args.goal})
                    if args.execute:
                        sock.sendto(wire, ('192.168.168.110', 5002))
                    sent_at = time.monotonic()
                    end = sent_at + 180
                    record('navigation_sent' if args.execute else 'navigation_prepared',
                           floor=target, topic='active_result', command=1, value=args.goal,
                           sequence=sequence, elapsed_after_floor_sec=sent_at-(deadline-10),
                           payload_hex=wire.hex(), motion_command_sent=args.execute)
                try:
                    wire, source = sock.recvfrom(65535)
                except socket.timeout:
                    continue
                if source[0] != '192.168.168.110':
                    record('rejected_source', source=list(source)); continue
                try:
                    packet = decode_packet(wire, source)
                except ValueError as error:
                    record('unparsed', source=list(source), error=str(error), payload_hex=wire.hex()); continue
                record('receive', source=list(source), topic=packet.topic,
                       sequence=packet.sequence, data=packet.data, payload_hex=wire.hex())
                if packet.topic != 'bt_control' or not isinstance(packet.data, dict):
                    continue
                command = packet.data.get('command'); value = packet.data.get('value')
                if command == 10:
                    if not isinstance(value, str) or value not in {str(i) for i in range(1,17)}:
                        record('rejected_floor', value=value); continue
                    if target is None:
                        target = value; deadline = time.monotonic()+10
                        record('floor_accepted', floor=target, delay_sec=10)
                    elif value != target:
                        record('busy_different_floor', floor=value); continue
                    else:
                        record('duplicate_floor_ignored', floor=target)
                    sock.sendto(b'accepted', source)
                    record('transport_ack', source=list(source))
                elif command == 1 and value == 'arrive':
                    if sent_at is None:
                        record('early_arrival_ignored'); continue
                    sock.sendto(b'accepted', source)
                    record('vendor_arrival_received', floor=target, goal=args.goal,
                           navigation_elapsed_sec=time.monotonic()-sent_at,
                           arm_started=False, verified_stationary=False)
                    finished=True; break
        finally:
            sock.close()
        record('test_finished', vendor_arrival_received=finished, floor=target,
               motion_command_sent=bool(args.execute and sent_at is not None), arm_started=False)
    return 0 if finished else 2


if __name__ == '__main__':
    raise SystemExit(main())
