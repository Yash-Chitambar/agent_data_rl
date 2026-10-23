#!/usr/bin/env python3
"""Multi-GPU job runner for a plain Linux box (no scheduler).

Reads one or more manifests (one shell command per line, same format as runs/queue/jobs.txt;
blank lines and lines starting with '#' are ignored) and runs each job with
CUDA_VISIBLE_DEVICES set to one GPU, `--per-gpu` jobs per GPU at a time.

Optional trailing annotations after ' # ' (bash ignores them, so lines stay valid for
scripts/tools/local_queue.sh):
    after=NAME[,NAME...]   start only after these runs are complete (runs/NAME/.complete)
    visual                 informational (job renders; counted separately in --status)
    mem=GB                 informational GPU memory estimate

State (default runs/queue_server/):
    done.txt        commands finished with exit 0 (the command text, without annotations)
    failed.txt      commands that failed after all retries (tab: rc, command)
    runner.log      event log
    jobs/           one stdout/stderr log per attempt; rc/ exit codes; running.json
A training job (has `--name X`) that exits 0 also gets runs/X/.complete (JSON). Jobs listed in
done.txt or whose run has .complete are skipped (use --force to rerun). A failed job is retried
`--retries` times (default 1). A training job whose run dir exists without .complete is started
with `--resume` appended (train.py / train_visual.py continue from runs/X/last.pt); the manifest
line is unchanged, so running jobs still match by command text. A partial dir without last.pt is
renamed to runs/X.failed-<time> and the run starts from scratch.

Manifests are re-read every poll, so lines appended while the runner is active are picked up.
The runner ignores SIGHUP and starts every job in its own session, but use tmux or nohup:
    tmux new -s jobs 'python3 scripts/server/run_jobs.py scripts/server/jobs/B_train.txt'
    nohup python3 scripts/server/run_jobs.py MANIFEST > runs/queue_server/nohup.out 2>&1 &
Ctrl-C / SIGTERM stops the runner AND its jobs (they are not marked done). If the runner dies
hard (kill -9, reboot of the runner only), a restarted runner adopts still-running jobs via
jobs/running.json and reads their exit codes from jobs/rc/.

Examples:
    run_jobs.py scripts/server/jobs/B_train.txt scripts/server/jobs/B_eval.txt --gpus 0,1,2,3
    run_jobs.py MANIFEST --per-gpu 2 --dry-run
    run_jobs.py MANIFEST --status
    run_jobs.py --mark-complete B-peg-proprio-nocontact-s0     # run trained elsewhere
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
NAME_RE = re.compile(r"--name[ =](\S+)")
REDIRECT_RE = re.compile(r">\s*(\S+\.log)")
TRAIN_RE = re.compile(r"blind_insertion\.train(_visual)?\b")


def now():
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@dataclass
class Job:
    idx: int                 # position over all manifests (priority order)
    manifest: str
    lineno: int
    cmd: str                 # command without annotations: the job's identity
    name: str | None
    after: list[str] = field(default_factory=list)
    tags: dict = field(default_factory=dict)

    @property
    def is_train(self):
        return self.name is not None and TRAIN_RE.search(self.cmd) is not None

    @property
    def label(self):
        return self.name or f"{Path(self.manifest).stem}:{self.lineno}"


_WARNED: set[str] = set()


def _warn_once(msg):
    if msg not in _WARNED:
        _WARNED.add(msg)
        print(f"[run_jobs] warning: {msg}", file=sys.stderr)


def parse_manifests(paths) -> list[Job]:
    jobs, seen = [], {}
    for p in paths:
        for lineno, raw in enumerate(Path(p).read_text().splitlines(), 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            cmd, _, ann = line.partition(" # ")
            cmd = cmd.strip()
            tags, after = {}, []
            for tok in ann.split():
                k, eq, v = tok.partition("=")
                if k == "after" and eq:
                    after += [x for x in v.split(",") if x]
                else:
                    tags[k] = v if eq else True
            if cmd in seen:
                _warn_once(f"duplicate command {p}:{lineno} (first at {seen[cmd]}); keeping the first")
                continue
            seen[cmd] = f"{p}:{lineno}"
            m = NAME_RE.search(cmd)
            jobs.append(Job(len(jobs), str(p), lineno, cmd, m.group(1) if m else None, after, tags))
    names = [j.name for j in jobs if j.is_train]
    for n in {n for n in names if names.count(n) > 1}:
        _warn_once(f"run name {n} appears in more than one training job")
    return jobs


def read_lines(p: Path) -> set[str]:
    return set(p.read_text().splitlines()) if p.exists() else set()


def detect_gpus() -> list[str]:
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd:
        return [g.strip() for g in cvd.split(",") if g.strip()]
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                                      text=True, timeout=30)
        return [x.strip() for x in out.splitlines() if x.strip()]
    except Exception as e:  # noqa: BLE001
        sys.exit(f"[run_jobs] cannot detect GPUs ({e}); pass --gpus")


def gpu_idle(gpu: str, samples=3, interval=1.0) -> tuple[bool, str]:
    """Fail closed if a named GPU is busy or its capacity cannot be established."""
    for sample in range(samples):
        try:
            capacity = subprocess.check_output(
                ["nvidia-smi", "-i", str(gpu),
                 "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                text=True, timeout=30).strip()
            memory, utilization = (float(x.strip()) for x in capacity.split(","))
            processes = subprocess.check_output(
                ["nvidia-smi", "-i", str(gpu), "--query-compute-apps=pid",
                 "--format=csv,noheader,nounits"], text=True, timeout=30).strip()
            if not (0 <= memory < 1024 and 0 <= utilization <= 1) or processes:
                return False, f"{memory:g} MiB, {utilization:g}% compute, pids={processes or 'none'}"
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            return False, f"capacity check failed: {exc}"
        if sample + 1 < samples:
            time.sleep(interval)
    return True, f"idle over {samples} sample(s)"


def with_resume(cmd: str) -> str:
    """Insert --resume before the first output redirection (or append it)."""
    m = re.search(r"\s(\d?>|&>)", cmd)
    return f"{cmd[:m.start()]} --resume{cmd[m.start():]}" if m else f"{cmd} --resume"


class Runner:
    def __init__(self, args):
        self.a = args
        self.state = Path(args.state_dir)
        self.runs = Path(args.runs_dir)
        self.jobdir = self.state / "jobs"
        self.rcdir = self.jobdir / "rc"
        for d in (self.state, self.jobdir, self.rcdir):
            d.mkdir(parents=True, exist_ok=True)
        self.done_f, self.failed_f = self.state / "done.txt", self.state / "failed.txt"
        self.running_f = self.jobdir / "running.json"
        self.log_f = self.state / "runner.log"
        gpus = args.gpus.split(",") if args.gpus not in ("", "auto") else detect_gpus()
        self.slots = [(g, k) for k in range(args.per_gpu) for g in gpus]   # spread over GPUs first
        self.running: dict[tuple, dict] = {}     # slot -> info (Popen or adopted pid)
        self.attempts: dict[str, int] = {}
        self.perm_failed: dict[str, int] = {}
        for line in read_lines(self.failed_f):
            if "\t" in line:
                rc, cmd = line.split("\t", 1)
                self.perm_failed[cmd] = int(rc) if rc.lstrip("-").isdigit() else 1
        self.stopping = False

    # ---------------------------------------------------------------- bookkeeping
    def log(self, msg):
        line = f"[{now()}] {msg}"
        print(line, flush=True)
        if not self.a.dry_run:
            with open(self.log_f, "a") as f:
                f.write(line + "\n")

    def complete(self, name) -> bool:
        return (self.runs / name / ".complete").exists()

    def mark_complete(self, name, info):
        d = self.runs / name
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / ".complete.tmp"
        tmp.write_text(json.dumps(info, indent=1))
        tmp.replace(d / ".complete")

    def save_running(self):
        data = [dict(slot=list(s), **{k: v for k, v in i.items() if k != "proc"})
                for s, i in self.running.items()]
        tmp = self.running_f.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=1))
        tmp.replace(self.running_f)

    def job_state(self, j: Job, done: set[str]) -> str:
        if any(i["cmd"] == j.cmd for i in self.running.values()):
            return "running"
        if j.cmd in done:
            return "done"
        if j.is_train and self.complete(j.name) and not self.a.force:
            return "complete"
        if j.cmd in self.perm_failed:
            return "failed"
        for n in j.after:
            if not self.complete(n):
                dep = next((x for x in self.jobs if x.name == n and x.is_train), None)
                if dep is not None and dep.cmd in self.perm_failed:
                    return "blocked"
                return "waiting"
        return "ready"

    # ---------------------------------------------------------------- process control
    def adopt(self):
        """Re-attach to jobs started by a previous runner that are still alive."""
        if not self.running_f.exists():
            return
        try:
            old = json.loads(self.running_f.read_text())
        except Exception:  # noqa: BLE001
            return
        for info in old:
            slot = tuple(info.pop("slot"))
            pid = info["pid"]
            if Path(f"/proc/{pid}").exists():
                self.running[slot] = info      # tracked even if its GPU is not in --gpus now
                self.log(f"ADOPT pid {pid} on GPU {slot[0]}: {info['label']}")
            else:
                self.running[slot] = info      # finished while we were away; reaped below
                info["gone"] = True
        self.reap()

    def start(self, j: Job, slot):
        gpu = slot[0]
        if self.a.require_idle_gpus:
            idle, reason = gpu_idle(gpu)
            if not idle:
                self.log(f"WAIT gpu={gpu}: {reason}")
                return False
            # Recheck immediately before changing run state and launching the job.
            idle, reason = gpu_idle(gpu, samples=1)
            if not idle:
                self.log(f"WAIT gpu={gpu}: {reason}")
                return False
        n = self.attempts.get(j.cmd, 0) + 1
        self.attempts[j.cmd] = n
        cmd = j.cmd
        if j.is_train:
            d = self.runs / j.name
            if d.exists() and not self.complete(j.name):
                if (d / "last.pt").exists() and "--resume" not in cmd:
                    cmd = with_resume(cmd)
                    self.log(f"partial run dir {d} has last.pt: resuming")
                elif not (d / "last.pt").exists():
                    bak = self.runs / f"{j.name}.failed-{dt.datetime.now():%Y%m%d-%H%M%S}"
                    self.log(f"partial run dir {d} (no .complete, no last.pt) -> {bak}")
                    if not self.a.dry_run:
                        d.rename(bak)
        slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", j.label)[:80]
        logp = self.jobdir / f"{j.idx:04d}-{slug}-a{n}.log"
        rcp = self.rcdir / f"{j.idx:04d}-{slug}-a{n}.rc"
        self.log(f"START gpu={gpu} attempt={n} {j.label}: {j.cmd}")
        if self.a.dry_run:
            return
        rcp.unlink(missing_ok=True)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), RUN_JOBS_GPU=str(gpu))
        wrapper = 'source scripts/server/env.sh; bash -c "$1"; rc=$?; echo $rc > "$2"; exit $rc'
        with open(logp, "w") as lf:
            lf.write(f"# {now()} host={socket.gethostname()} gpu={gpu} attempt={n}\n# {j.cmd}\n")
            lf.flush()
            p = subprocess.Popen(["bash", "-c", wrapper, "job", cmd, str(rcp)], cwd=REPO, env=env,
                                 stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
        self.running[slot] = dict(proc=p, pid=p.pid, cmd=j.cmd, label=j.label, name=j.name,
                                  is_train=j.is_train, gpu=gpu, attempt=n, t0=time.time(),
                                  start=now(), log=str(logp), rc_file=str(rcp))
        self.save_running()
        return True

    def reap(self):
        for slot, info in list(self.running.items()):
            p = info.get("proc")
            if p is not None:
                rc = p.poll()
                if rc is None:
                    continue
            else:   # adopted
                if not info.get("gone") and Path(f"/proc/{info['pid']}").exists():
                    continue
                rcp = Path(info["rc_file"])
                rc = int(rcp.read_text().strip()) if rcp.exists() else -1
            del self.running[slot]
            self.finish(info, rc)
        self.save_running()

    def finish(self, info, rc):
        mins = (time.time() - info["t0"]) / 60 if "t0" in info else float("nan")
        cmd = info["cmd"]
        if rc == 0:
            self.log(f"END ok gpu={info['gpu']} {mins:.1f} min {info['label']}")
            with open(self.done_f, "a") as f:
                f.write(cmd + "\n")
            if info.get("is_train") and info.get("name"):
                self.mark_complete(info["name"], dict(cmd=cmd, rc=0, host=socket.gethostname(),
                                                      gpu=info["gpu"], start=info.get("start"),
                                                      end=now(), minutes=round(mins, 1), by="run_jobs"))
            return
        self.log(f"END FAIL rc={rc} gpu={info['gpu']} {mins:.1f} min {info['label']} (log {info['log']})")
        m = REDIRECT_RE.search(cmd)
        if m and (REPO / m.group(1)).exists():
            tail = (REPO / m.group(1)).read_text(errors="replace").splitlines()[-15:]
            self.log("  tail of " + m.group(1) + ":\n    " + "\n    ".join(tail))
        if self.stopping:
            return
        if self.attempts.get(cmd, 1) > self.a.retries:
            self.perm_failed[cmd] = rc
            with open(self.failed_f, "a") as f:
                f.write(f"{rc}\t{cmd}\n")
            self.log(f"GIVE UP after {self.attempts.get(cmd, 1)} attempts: {info['label']}")
        else:
            self.log(f"RETRY queued: {info['label']}")

    def stop(self, signum, _frame):
        if self.stopping:
            return
        self.stopping = True
        self.log(f"signal {signum}: stopping {len(self.running)} job(s)")
        for info in self.running.values():
            try:
                os.killpg(info["pid"], signal.SIGTERM)
            except ProcessLookupError:
                pass
        t = time.time()
        while self.running and time.time() - t < 30:
            time.sleep(1)
            self.reap()
        for info in self.running.values():
            try:
                os.killpg(info["pid"], signal.SIGKILL)
            except ProcessLookupError:
                pass
        self.running.clear()
        self.save_running()
        sys.exit(130)

    # ---------------------------------------------------------------- main loop
    def run(self):
        self.jobs = parse_manifests(self.a.manifests)
        self.adopt()
        self.log(f"runner pid {os.getpid()} on {socket.gethostname()}: {len(self.jobs)} jobs, "
                 f"slots {[s[0] for s in self.slots]} (per GPU {self.a.per_gpu}), retries {self.a.retries}"
                 + (" [DRY RUN]" if self.a.dry_run else ""))
        started = 0
        while True:
            self.jobs = parse_manifests(self.a.manifests)
            self.reap()
            done = read_lines(self.done_f)
            states = {j.idx: self.job_state(j, done) for j in self.jobs}
            ready = [j for j in self.jobs if states[j.idx] == "ready"]
            if self.a.only:
                ready = [j for j in ready if re.search(self.a.only, j.cmd)]
            if self.a.skip_tag:
                ready = [j for j in ready if not any(t in j.tags for t in self.a.skip_tag)]
            free = [s for s in self.slots if s not in self.running]
            if self.a.max_jobs:
                ready = ready[:max(self.a.max_jobs - started, 0)]
            if self.a.dry_run:
                for jb, slot in zip(ready, free):
                    print(f"would START gpu={slot[0]} {jb.label}: {jb.cmd}")
                self.print_status(done, header="dry-run (state before starting)")
                return 0
            waiting_for_gpu = False
            for slot in free:
                if not ready:
                    break
                if self.start(ready[0], slot):
                    ready.pop(0)
                    started += 1
                else:
                    waiting_for_gpu = True
            if not self.running and not self.a.watch and not waiting_for_gpu:
                waiting = [jb for jb in self.jobs if states[jb.idx] == "waiting"]
                if waiting:
                    missing = sorted({n for jb in waiting for n in jb.after if not self.complete(n)})
                    self.log(f"{len(waiting)} job(s) wait on runs that are not complete and not produced "
                             f"by these manifests: {', '.join(missing)} (see --mark-complete)")
                self.print_status(read_lines(self.done_f))
                nfail = sum(1 for jb in self.jobs if jb.cmd in self.perm_failed)
                self.log(f"runner finished ({nfail} failed, {len(waiting)} waiting)")
                return 1 if nfail or waiting else 0
            time.sleep(self.a.poll)

    def print_status(self, done, header="status"):
        self.jobs = getattr(self, "jobs", None) or parse_manifests(self.a.manifests)
        rows = [(self.job_state(j, done), j) for j in self.jobs]
        counts = {}
        for s, _ in rows:
            counts[s] = counts.get(s, 0) + 1
        print(f"--- {header}: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
        for s, j in rows:
            gpu = next((i["gpu"] for i in self.running.values() if i["cmd"] == j.cmd), "")
            tag = " [visual]" if j.tags.get("visual") else ""
            print(f"  {s:9s} {('gpu' + str(gpu)) if gpu != '' else '':5s} {j.label}{tag}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("manifests", nargs="*")
    ap.add_argument("--gpus", default="auto", help="comma list of GPU ids, or auto (CUDA_VISIBLE_DEVICES / nvidia-smi)")
    ap.add_argument("--per-gpu", type=int, default=1, help="concurrent jobs per GPU")
    ap.add_argument("--require-idle-gpus", action="store_true",
                    help="before each launch require <1 GiB, <=1%% compute and no compute PIDs "
                         "over repeated samples; wait when busy (requires --per-gpu 1)")
    ap.add_argument("--retries", type=int, default=1)
    ap.add_argument("--state-dir", default=str(REPO / "runs" / "queue_server"))
    ap.add_argument("--runs-dir", default=str(REPO / "runs"))
    ap.add_argument("--poll", type=float, default=10.0)
    ap.add_argument("--watch", action="store_true", help="keep polling for appended lines when idle")
    ap.add_argument("--only", default="", help="regex: run only matching commands")
    ap.add_argument("--skip-tag", action="append", default=[], metavar="TAG",
                    help="never start jobs carrying this annotation tag (repeatable), e.g. "
                         "--skip-tag also_in_local_queue --skip-tag needs_local_ckpt")
    ap.add_argument("--max-jobs", type=int, default=0, help="start at most this many jobs, then drain")
    ap.add_argument("--force", action="store_true", help="ignore runs/<name>/.complete markers")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--mark-complete", nargs="+", metavar="NAME",
                    help="write runs/NAME/.complete for runs finished elsewhere, then exit")
    a = ap.parse_args()
    if a.require_idle_gpus and a.per_gpu != 1:
        ap.error("--require-idle-gpus requires --per-gpu 1")
    if a.mark_complete:
        r = Runner(argparse.Namespace(**{**vars(a), "gpus": "0", "per_gpu": 1}))
        for n in a.mark_complete:
            if not (r.runs / n / "last.pt").exists():
                print(f"warning: {r.runs / n}/last.pt does not exist", file=sys.stderr)
            r.mark_complete(n, dict(by="mark-complete", host=socket.gethostname(), end=now()))
            print(f"marked {n} complete")
        return 0
    if not a.manifests:
        ap.error("no manifest given")
    if a.status:
        r = Runner(argparse.Namespace(**{**vars(a), "gpus": a.gpus if a.gpus != "auto" else "0"}))
        if r.running_f.exists():
            for info in json.loads(r.running_f.read_text()):
                if Path(f"/proc/{info['pid']}").exists():
                    r.running[tuple(info.pop("slot"))] = info
        r.jobs = parse_manifests(a.manifests)
        r.print_status(read_lines(r.done_f))
        return 0
    r = Runner(a)
    lock = open(r.state / "runner.lock", "w")
    if not a.dry_run:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            sys.exit(f"[run_jobs] another runner holds {r.state}/runner.lock")
        lock.write(f"{os.getpid()} {socket.gethostname()}\n")
        lock.flush()
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    signal.signal(signal.SIGINT, r.stop)
    signal.signal(signal.SIGTERM, r.stop)
    return r.run()


if __name__ == "__main__":
    sys.exit(main())
