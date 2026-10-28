import importlib.util
from pathlib import Path
import json
import pytest
s=importlib.util.spec_from_file_location('prep',Path(__file__).parents[1]/'scripts/real/prepare_bimanual_session.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)


def test_offline_preparation_and_no_overwrite(tmp_path):
    p=tmp_path/'session';m.prepare(p)
    x=json.loads((p/'session.json').read_text())
    assert {t['side'] for t in x['trials'] if t.get('kind')=='pc_arm_contact'}=={'left','right'}
    assert x['commissioning_status']=='offline_prepared_only'
    with pytest.raises(ValueError):m.prepare(p)


def test_timeline_preserves_clock_and_source(tmp_path):
    (tmp_path/'robot.csv').write_text('t_tick,side\n2.0,left\n')
    (tmp_path/'camera.csv').write_text('host_receive_ns,camera_image_ns\n1000000000,99\n')
    out=tmp_path/'timeline.jsonl';r=m.timeline(tmp_path,out)
    rows=[json.loads(t) for t in out.read_text().splitlines()]
    assert r['events']==2
    assert [e['host_monotonic_ns'] for e in rows]==[1000000000,2000000000]
    assert rows[0]['record']['camera_image_ns']=='99'
    with pytest.raises(ValueError):m.timeline(tmp_path,out)
