"""Build consistent report inputs with unrestricted NBA totals, reusing valid replays.

The primary cohort overlays cached block-off NBA replays on the original cohort.
Changed model order times invalidate model-time baselines; guard-off comparisons
also disable the totals block. No cached result or source recording is deleted.
Run --refresh to complete required local replays before rendering the thesis.
"""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
OUT = HERE / 'out'
PRIMARY = 'b68_main_all_totals'
BASELINES = 'b69_baselines_all_totals'
NO_GUARD = 'b70_noguard_all_totals'
GUARDS = ('skip: feed', 'skip: ball in play', 's since the last known pitch',
          'signal differs on real-time stamps', 'skip: review', 'skip: placeholder')


def records(name):
    out = {}
    for p in sorted((OUT/name/'games').glob('*/*/result.json')):
        r = json.loads(p.read_text())
        if r.get('results'):
            out[(r['league'],r['game'])] = (p.parent,r)
    return out


def profile(folder):
    orders, late, guard = [], False, False
    with gzip.open(folder/'host/decisions.jsonl.gz','rb') as handle:
        for line in handle:
            if b'"order"' not in line and b'skip:' not in line:
                continue
            item = json.loads(line)
            for row in item if isinstance(item,list) else [item]:
                action = row.get('action') or ''
                if action == 'order':
                    orders.append((row['ticker'],round(row['t'],1)))
                late |= action.startswith('skip: late game')
                guard |= any(token in action for token in GUARDS)
    digest = hashlib.sha256(json.dumps(sorted(set(orders))).encode()).hexdigest()
    return dict(digest=digest,late=late,guard=guard)


def link(name, key, source, record):
    dest = OUT/name/'games'/record['date']/f'{key[0]}_{key[1]}'
    dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.is_symlink():
        if dest.resolve() == source.resolve():
            return
        dest.unlink()
    if dest.exists():
        assert dest.resolve() == source.resolve(), f'Refusing to replace real replay: {dest}'
        return
    dest.symlink_to(source.resolve(),target_is_directory=True)


def build():
    original, no_late = records('b55_main_20x'), records('b57_nolate')
    original_guard, combined = records('b57_noguard'), records(NO_GUARD)
    original_base, updated_base = records('b60_baselines'), records(BASELINES)
    jobs = dict(nolate=[],noguard_nolate=[],baselines=[])
    selected, profiles = {}, {}
    for key,(folder,r) in original.items():
        old = profile(folder)
        profiles[key] = old
        choice = (folder,r)
        if key[0]=='NBA' and old['late']:
            if key not in no_late or no_late[key][1]['rule_hash'] != r['rule_hash']:
                jobs['nolate'].append(key[1]); continue
            choice = no_late[key]
        selected[key] = choice
        link(PRIMARY,key,*choice)
        if old['guard']:
            prior = original_guard.get(key)
            if prior and prior[1]['rule_hash']==r['rule_hash'] and (key[0]!='NBA' or not profile(prior[0])['late']):
                link(NO_GUARD,key,*prior)
            elif key not in combined or combined[key][0].is_symlink() or combined[key][1].get('variant') != 'noguard_nolate':
                dest = OUT/NO_GUARD/'games'/r['date']/f'{key[0]}_{key[1]}'
                if dest.is_symlink(): dest.unlink()
                jobs['noguard_nolate'].append(key[1])
        # Preserve the established matched-game cohort: additional baseline games
        # are not selected by their outcomes or by availability of late trades.
        if key not in original_base:
            continue
        now = profile(choice[0]) if choice[0] != folder else old
        if now['digest'] == old['digest']:
            link(BASELINES,key,*original_base[key])
        elif (key not in updated_base or updated_base[key][0].is_symlink() or
              updated_base[key][1].get('model_times_sha256') != now['digest'] or
              updated_base[key][1].get('baseline_code_sha256') != hashlib.sha256((HERE/'b60_baseline_replay.py').read_bytes()).hexdigest()):
            dest=OUT/BASELINES/'games'/r['date']/f'{key[0]}_{key[1]}'
            if dest.is_symlink(): dest.unlink()
            jobs['baselines'].append(key[1])
    (OUT/PRIMARY).mkdir(exist_ok=True)
    (OUT/PRIMARY/'report_inputs.json').write_text(json.dumps(dict(jobs=jobs,
        cohort=[dict(league=k[0],game=k[1],source=str(v[0])) for k,v in selected.items()]),indent=2))
    return jobs


def refresh():
    while True:
        jobs=build()
        print({key:len(value) for key,value in jobs.items()},flush=True)
        if not any(jobs.values()): return
        if shutil.disk_usage(OUT).free < 8*1024**3:
            raise RuntimeError('Insufficient free disk for local replay')
        for variant in ('nolate','noguard_nolate','baselines'):
            if not jobs[variant]: continue
            env=os.environ.copy()
            env.update(GAMES=','.join(jobs[variant]),ARMS='sym_model,micro_model,flow_model' if variant=='baselines' else 'host',
                       WORKERS=env.get('WORKERS','2'),SIZING='20x',OMP_NUM_THREADS='1')
            if variant=='baselines':
                env.update(OUT_NAME=BASELINES,MAIN_NAME=PRIMARY)
                script='b60_baseline_replay.py'
            else:
                env.update(OUT_NAME='b57_nolate' if variant=='nolate' else NO_GUARD,VARIANT=variant)
                script='b57_variant_replay.py'
            subprocess.run([sys.executable,str(HERE/script),'--local-only'],env=env,check=True,cwd=HERE.parents[1])
            # Guard variants include metadata added by this wrapper for cache audit.
            if variant=='noguard_nolate':
                for _,(folder,r) in records(NO_GUARD).items():
                    if not folder.is_symlink() and r['game'] in jobs[variant]:
                        r['variant']=variant
                        (folder/'result.json').write_text(json.dumps(r,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--refresh',action='store_true')
    args=parser.parse_args()
    if args.refresh: refresh()
    else: print(json.dumps(build(),indent=2))
