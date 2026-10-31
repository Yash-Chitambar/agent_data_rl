"""Summarize existing cross-pose replay reports without altering raw captures."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
root=Path('results/usbc/bimanual_20261029/replay')
groups={
 'Right pickup':{'baseline':'driver_cross_pose/right_pickup_baseline','driver':'driver_cross_pose/right_pickup_recorded','driver + friction':'driver_cross_pose/right_pickup_recorded_friction'},
 'Right folded rest J2':{'baseline':'right_j2_baseline','driver':'driver_cross_pose/right_rest_j2_recorded','driver + friction':'driver_cross_pose/right_rest_j2_recorded_friction'},
 'Left elevated':{'baseline':'free_settle_001_baseline','driver':'driver_cross_pose/free_settle_001_driver_reconstruction'}}
report={'scope':'offline encoder replay, contacts disabled; no external wrist accuracy',
        'decision':'do not adopt combined driver/friction candidate as default; cross-pose regressions remain',
        'trials':{}}
fig,axes=plt.subplots(1,3,figsize=(15,4))
for ax,(trial,variants) in zip(axes,groups.items()):
    report['trials'][trial]={}
    for k,(label,stem) in enumerate(variants.items()):
        p=root/(stem+'.json');r=json.loads(p.read_text())
        values=np.array(r['joint_rmse_rad'])*1000
        ax.bar(np.arange(6)+k*.25,values,.24,label=label)
        report['trials'][trial][label]={'source_report':str(p),'joint_rmse_mrad':values.tolist()}
    ax.set(title=trial,xlabel='Joint',ylabel='Encoder RMSE (mrad)',xticks=np.arange(6)+.25,xticklabels=np.arange(1,7))
    ax.legend(fontsize=8)
fig.suptitle('Recorded-state driver reconstruction: improvement is not consistent across poses')
fig.tight_layout()
dest=root/'driver_cross_pose'
fig.savefig(dest/'comparison.png',dpi=140)
(dest/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
