"""Replay measured host-side camera cadence with delayed publication and held RGB.

Capture at host-receive time is an explicit proxy: sensor exposure age is unknown.
The initial/reset frame assumes a stationary pre-roll at the reset pose.
"""
import json
from pathlib import Path
import torch


class CameraCadence:
    def __init__(self, render, nworld, device, profile=None):
        source=profile or Path(__file__).resolve().parents[1]/'configs/bimanual_camera_cadence.json'
        data=json.loads(Path(source).read_text())
        if data['schema']!='host-camera-cadence-v1':raise ValueError('unsupported camera cadence schema')
        self.schedule=data['cameras'];self.render=render;self.device=device
        self.cursor={name:0 for name in self.schedule};self.cycle={name:0 for name in self.schedule}
        self.pending=[];self.generation=torch.zeros(nworld,device=device,dtype=torch.long)
        self.frames={};self.captured={};self.published_index={}
        for name,s in self.schedule.items():
            if not s['capture_s'] or len(s['capture_s'])!=len(s['ready_s']):raise ValueError('invalid camera timeline')
            if any(b<=a for a,b in zip(s['capture_s'],s['capture_s'][1:])):raise ValueError('unordered captures')
            if any(b<a for a,b in zip(s['capture_s'],s['ready_s'])):raise ValueError('publication precedes capture')
            if s['cycle_s']<=s['capture_s'][-1]:raise ValueError('invalid cycle duration')
            self.captured[name]=torch.zeros(nworld,device=device)
            self.published_index[name]=torch.full((nworld,),-1,device=device,dtype=torch.long)

    def reset(self, ids, time):
        self.generation[ids]+=1
        frames=self.render()
        for name,frame in frames.items():
            if name not in self.frames:self.frames[name]=frame.clone()
            else:self.frames[name][ids]=frame[ids]
            self.captured[name][ids]=time
            self.published_index[name][ids]=-1

    def advance(self,time):
        captures=[]
        for name,s in self.schedule.items():
            i=self.cursor[name];cycle=self.cycle[name];length=len(s['capture_s'])
            while cycle*s['cycle_s']+s['capture_s'][i]<=time+1e-10:
                captures.append((name,cycle*length+i,max(time,cycle*s['cycle_s']+s['ready_s'][i])))
                i+=1
                if i==length:i=0;cycle+=1
            self.cursor[name]=i;self.cycle[name]=cycle
        if captures:
            frames=self.render()
            for name,index,ready in captures:
                self.pending.append((ready,name,index,time,frames[name].clone(),self.generation.clone()))
        pending=[]
        for ready,name,index,captured,frame,generation in self.pending:
            if ready>time+1e-10:
                pending.append((ready,name,index,captured,frame,generation));continue
            # Never publish a pre-reset frame or regress to an older late frame.
            valid=(generation==self.generation)&(index>self.published_index[name])
            self.frames[name][valid]=frame[valid]
            self.captured[name][valid]=captured
            self.published_index[name][valid]=index
        self.pending=pending

    def observation(self,time):
        return {name:frame.clone() for name,frame in self.frames.items()}, {
            name:(time-t).clamp(min=0).clone() for name,t in self.captured.items()}
