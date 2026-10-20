#!/usr/bin/env bash
# Sequential GPU job queue: runs each line of runs/queue/jobs.txt not yet marked done.
# Append jobs with: echo "<cmd>" >> runs/queue/jobs.txt
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD MUJOCO_GL=egl
Q=runs/queue
touch $Q/jobs.txt $Q/done.txt
while true; do
  next=$(grep -vxF -f $Q/done.txt $Q/jobs.txt | head -1)
  if [ -z "$next" ]; then sleep 60; continue; fi
  # wait for any running trainer started outside the queue
  while pgrep -f "blind_insertion.train" > /dev/null; do sleep 30; done
  echo "[$(date)] START $next" >> $Q/queue.log
  bash -c "$next" >> $Q/queue.log 2>&1
  echo "[$(date)] END rc=$? $next" >> $Q/queue.log
  echo "$next" >> $Q/done.txt
done
