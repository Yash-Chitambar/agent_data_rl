"""Merge LeRobot dataset shards written by pick_to_lerobot.py --shard i --nshards n (LeRobot venv).

    .venv-act/bin/python scripts/policy/pick_merge_shards.py --shards data/act/pick_v3_s{0..7} --root data/act/pick_v3
"""
import argparse
from pathlib import Path
from lerobot.datasets.aggregate import aggregate_datasets

ap = argparse.ArgumentParser()
ap.add_argument('--shards', nargs='+', required=True)
ap.add_argument('--root', required=True)
a = ap.parse_args()
roots = [Path(s) for s in a.shards]
aggregate_datasets(repo_ids=[f'local/{r.name}' for r in roots], aggr_repo_id=f'local/{Path(a.root).name}', roots=roots, aggr_root=Path(a.root))
print('merged', len(roots), 'shards ->', a.root)
