"""Run the real pointing entry with every hardware/vision boundary replaced."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import time

import numpy as np
import pytest
import yaml

import run_online_replan_no_contact as entry
from piper_handeye.kinematics import forward_kinematics


@pytest.mark.parametrize('scenario',['same_target','stale_target','other_reference','acquisition_failed',
                                     'local_context_anchor_failed'])
def test_real_entry_rejoins_normal_pointing_without_search_motion(tmp_path,monkeypatch,scenario):
    origin=np.zeros(6)
    config=dict(robot={},safety_stop={'host':'unused','port':1,'token':'unused'},
                vision_service={'host':'unused','port':1,'token':'unused'},
                timed_trajectory={'host':'unused','port':1,'token':'unused'},
                depth={'capture_size':[640,360]},calibration={'handeye_yaml':'unused'},
                detector={'target_confidence':.70})
    cfg=tmp_path/'config.yaml';cfg.write_text(yaml.safe_dump(config))
    hold=tmp_path/'hold.yaml';hold.write_text(yaml.safe_dump(dict(
        stage_1_states=[{'joints_rad':origin.tolist()}],stage_1=dict(
            planned_joints_rad=[origin.tolist()],selected_approach_roll_deg=0,
            base_to_button=np.eye(4).tolist()))))
    monkeypatch.setattr(entry,'ROOT',tmp_path)
    monkeypatch.setenv('YOUXIAOMEI_MISSION_STATUS_OUTBOX',str(tmp_path/'status.jsonl'))
    monkeypatch.setenv('YOUXIAOMEI_TASK_ID','isolated-test')
    monkeypatch.setattr(entry,'load_handeye',lambda p:(np.eye(4),np.eye(3),np.zeros(5)))
    state=SimpleNamespace(joints=origin.copy(),raw={'hardware_ok':1},
                          base_to_gripper=forward_kinematics(origin),timestamp=time.monotonic())
    robot=Mock();robot.state.return_value=state
    monkeypatch.setattr(entry,'PiperUdpRobot',Mock(return_value=robot))
    vision=Mock();vision.request.return_value={'ok':True,'barrier_sequence':0}
    monkeypatch.setattr(entry,'VisionRpcClient',Mock(return_value=vision))
    trajectory=Mock();trajectory.execute.side_effect=AssertionError('unexpected arm motion')
    monkeypatch.setattr(entry,'TimedTrajectoryUdpClient',Mock(return_value=trajectory))
    stop=Mock();stop.reset_latch.side_effect=AssertionError('unexpected stop reset')
    stop.health.return_value={'ok':True,'latched':False}
    monkeypatch.setattr(entry,'SafetyStopUdpClient',Mock(return_value=stop))
    supervisor=Mock(side_effect=AssertionError('no search guard needed without search motion'))
    monkeypatch.setattr(entry,'VisualSafetySupervisor',supervisor)
    labels=[('1',300,130),('2',300,100),('4',300,40),
            ('9',340,100),('10',340,70),('11',340,40)]
    response=dict(image_size=[640,360],detections=[dict(label=l,confidence=.55 if l=='4' else .95,
                  bbox_xyxy=[x-10,y-12,x+10,y+12]) for l,x,y in labels])
    target = '4'
    if scenario=='local_context_anchor_failed':
        record=json.loads((Path(__file__).parent/'fixtures/numeric_search_context/recorded_7_missing.json').read_text())
        response=record['frames'][0]
        response['capture_time_unix_sec']=time.time()
        target='7'
    monkeypatch.setattr(entry,'wait_for_stable_visibility_after_target_switch',
                        Mock(return_value=(response,False)))
    if scenario=='other_reference':
        real=entry.predict_button_center_from_grid
        monkeypatch.setattr('piper_handeye.numeric_search_evidence.predict_button_center_from_grid',
                            lambda *a:SimpleNamespace(**{**vars(real(*a)),'anchor_label':'2'}))
    calls=[]
    def collect(*args,**kwargs):
        calls.append(str(args[3]))
        if scenario=='local_context_anchor_failed':
            raise RuntimeError('fresh neighbour depth anchor unavailable')
        if scenario=='acquisition_failed' and len(calls)>1:
            raise RuntimeError('no stable fresh button target before motion')
        stale=scenario in ('stale_target','acquisition_failed') and len(calls)==1
        samples=[dict(sequence=i,capture_time_unix_sec=time.time()-(1 if stale else .01),
                      accepted=True,confidence=.95,bbox_xyxy=[298,10,323,37]) for i in range(1,5)]
        samples[-1]['acquisition_summary']=dict(target_label=str(args[3]),window_sample_count=4)
        button=np.eye(4);button[0,3]=.5
        return SimpleNamespace(stable=True,base_to_button=button),samples
    monkeypatch.setattr(entry,'collect_stationary_target',collect)
    monkeypatch.setattr('sys.argv',['run_online_replan_no_contact.py','--config',str(cfg),
        '--hold-report',str(hold),'--target-label',target,'--receding-schedule',
        '--pointing-only','--hold-at-pointing','--visual-safety',
        '--execute','--acknowledge-clear-workspace'])
    if scenario=='local_context_anchor_failed':
        with pytest.raises(RuntimeError,match='fresh neighbour depth anchor unavailable'):
            entry.main()
        assert calls==['3']  # inferred 7 is never a depth or pressing target
    elif scenario=='acquisition_failed':
        with pytest.raises(RuntimeError,match='no stable fresh button target'):
            entry.main()
    else:
        assert entry.main()==0
        report=yaml.safe_load(next((tmp_path/'pointing_only_runs').glob('*/pointing_report.yaml')).read_text())
        assert report['status']=='pointing_complete_held_at_target_view'
        assert any(e['kind']=='numeric_target_recovered_without_search'
                   for e in report['auto_pointing']['outbound_events'])
        assert calls==(['4'] if scenario=='same_target' else ['2','4'] if scenario=='other_reference' else ['4','4'])
    trajectory.execute.assert_not_called();stop.reset_latch.assert_not_called()
    supervisor.assert_not_called()
