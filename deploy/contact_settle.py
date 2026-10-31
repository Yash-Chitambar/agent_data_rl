"""Bounded high-pose controller comparison; never called during contact."""
import time
import numpy as np
from deploy.approach_trim import ApproachTrim
from deploy import conventions as C


def characterize(arms,observations,guard,camera_health,rows,meta,goal,trim_seconds=12.):
    if not np.isfinite(trim_seconds) or not 1<=trim_seconds<=30:raise ValueError('settle duration must be 1..30 seconds')
    import estop,safety
    robot=C.Robot();off=C.joint_offsets('left');nominal=robot.fk(goal+off)[0].numpy()
    trim=ApproachTrim(ki=.3,limit=.025,rate=.003,deadband=.002)
    meta['settle_candidate']={'ki':.3,'limit_rad':.025,'rate_rad_s':.003,'deadband_rad':.002,
                              'trim_duration_s':trim_seconds,'status':'candidate only; not adopted','pose':'high free-space only'}
    for phase,seconds in [('baseline_hold',3.),('candidate_trim',trim_seconds),('candidate_frozen_hold',2.)]:
        print('Free-space controller phase:',phase,flush=True)
        deadline=time.monotonic();errors=[];joint_errors=[];bias=np.zeros(6) if phase=='baseline_hold' else trim.bias.copy()
        for k in range(round(seconds/.02)):
            if estop.stopped():raise safety.SafetyStop('software stop during free-space comparison')
            if k%5==0:camera_health()
            obs=observations();poses={s:v[0][:6] for s,v in obs.items()};guard(poses)
            measured=poses['left'];error=goal-measured
            if np.max(np.abs(error))>.15:raise safety.SafetyStop('free-space nominal tracking error')
            if phase=='candidate_trim':bias=trim.step(error,.02)
            target=goal+bias
            previous=arms['left'].safe.last_safe[:6].copy()
            for alpha in np.linspace(0,1,5):guard({'left':previous*(1-alpha)+target*alpha,'right':poses['right']})
            arms['right'].send(np.r_[arms['right'].safe.last_safe[:6],obs['right'][0][6]],len(rows))
            arms['left'].send(np.r_[target,.02],len(rows))
            cart=float(np.linalg.norm(robot.fk(measured+off)[0].numpy()-nominal))
            errors.append(cart);joint_errors.append(float(np.max(np.abs(error))))
            rows.append({'host_monotonic_ns':time.monotonic_ns(),'phase':phase,
                         'nominal_left_encoder':goal.tolist(),'bias_rad':bias.tolist(),
                         'requested_left_encoder':target.tolist(),
                         'executed_left_encoder':arms['left'].safe.last_safe.tolist(),
                         'model_fk_error_m':cart,'observations':{s:{'q_encoder':v[0].tolist(),
                         'qd_encoder':v[1].tolist(),'observation_time':v[2],'time_exact':v[3]} for s,v in obs.items()}})
            deadline+=.02;time.sleep(max(0,deadline-time.monotonic()))
        meta.setdefault('settle_results',{})[phase]={'final_model_fk_error_m':errors[-1],
            'last_second_max_model_fk_error_m':max(errors[-50:]),
            'last_second_max_joint_error_rad':max(joint_errors[-50:]),'bias_end_rad':bias.tolist()}
        print(phase,meta['settle_results'][phase],flush=True)
    frozen=meta['settle_results']['candidate_frozen_hold']
    meta['settle_candidate']['attainment_gate_passed']=bool(frozen['last_second_max_model_fk_error_m']<=.003 and frozen['last_second_max_joint_error_rad']<=.01)
    print('Comparison complete; returning to rest. Candidate remains unadopted.',flush=True)
