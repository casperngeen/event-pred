"""Run a shard of the b49 event replay on fyp-vm, then fetch back only the small per-game result files.

The VM replays games it recorded itself (data/sports_live/vm -> its own recordings), so no raw data crosses the network.
Code and weights are staged identical to the Mac's, so b49's rule_hash matches and results merge by copying.

  b56_vm_shard.py stage                      copy code + deploy weights to the VM workspace
  b56_vm_shard.py check                      import test on the VM
  b56_vm_shard.py run  OUT_NAME GAMES        launch detached (GAMES: comma-separated game ids); env ARMS, SIZING, WORKERS pass through
  b56_vm_shard.py fetch OUT_NAME [--wait]    copy result.json files into out/OUT_NAME on the Mac (--wait: until the remote run ends)
"""
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[2]
VM = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', 'fyp-vm']
WS = '/home/casper/replay_ws'
PY = '/home/casper/fyp/venv/bin/python'
CODE = ['analysis/graph_2026_10/' + n for n in ('b36_live_loss_audit.py', 'b37_replay_lookup.py', 'b37_standard_replay.py', 'b49_event_replay_v2.py')]


def sh(cmd, **kw):
    return subprocess.run(VM + [cmd], check=True, text=True, capture_output=True, **kw).stdout


def stage():
    sh(f'mkdir -p {WS}/data/sports_live && ln -sfn /home/casper/fyp/sports_live {WS}/data/sports_live/vm')
    child = subprocess.Popen(VM + [f'tar xzf - -C {WS}'], stdin=subprocess.PIPE)
    with tarfile.open(fileobj=child.stdin, mode='w|gz') as tar:
        for p in sorted((ROOT / 'scripts').glob('*.py')):
            tar.add(p, arcname=f'scripts/{p.name}')
        for p in sorted((ROOT / 'stg_infra').rglob('*.py')):
            tar.add(p, arcname=str(p.relative_to(ROOT)))
        for rel in CODE:
            tar.add(ROOT / rel, arcname=rel)
        for p in sorted((ROOT / 'results/ingame/deploy').glob('*')):
            tar.add(p, arcname=str(p.relative_to(ROOT)))
    child.stdin.close()
    assert child.wait() == 0
    print('staged', sh(f'du -sh {WS} | cut -f1').strip())


def check():
    print(sh(f'cd {WS}/analysis/graph_2026_10 && {PY} -c "import b49_event_replay_v2 as B; print(sorted(B.B.MODELS))" 2>&1 | tail -3'))


def run(out_name, games):
    env = ' '.join(f'{k}={shlex.quote(os.environ[k])}' for k in ('ARMS', 'SIZING') if k in os.environ)
    env += f' OUT_NAME={shlex.quote(out_name)} GAMES={shlex.quote(games)} OMP_NUM_THREADS=1 WORKERS={os.environ.get("WORKERS", "2")}'
    sh(f'cd {WS}/analysis/graph_2026_10 && rm -f out/{out_name}/status.json; '
       f'nohup env {env} {PY} -u b49_event_replay_v2.py --monitor --local-only > run_{out_name}.log 2>&1 < /dev/null & echo started')
    print('launched on fyp-vm', out_name)


def remote_state(out_name):
    return sh(f"cat {WS}/analysis/graph_2026_10/out/{out_name}/status.json 2>/dev/null || echo '{{}}'")


def fetch(out_name, wait):
    import json
    while wait:
        st = json.loads(remote_state(out_name) or '{}').get('state', '')
        print(time.strftime('%H:%M:%S'), st, flush=True)
        if st.startswith('completed') or st in ('failed_or_incomplete', 'preflight_failed'):
            break
        time.sleep(30)
    base = f'{WS}/analysis/graph_2026_10/out/{out_name}'
    dest = ROOT / 'analysis/graph_2026_10/out' / out_name
    dest.mkdir(parents=True, exist_ok=True)
    child = subprocess.Popen(VM + [f"cd {base} && find games -name result.json | tar czf - -T -"], stdout=subprocess.PIPE)
    n = 0
    with tarfile.open(fileobj=child.stdout, mode='r|gz') as tar:
        for m in tar:
            assert not m.name.startswith('/') and '..' not in Path(m.name).parts and m.isfile() and m.size < 20 * 1024 ** 2
            tar.extract(m, dest, filter='data')
            n += 1
    assert child.wait() == 0
    print('fetched', n, 'result files into', dest)


if __name__ == '__main__':
    c = sys.argv[1]
    if c == 'stage':
        stage()
    elif c == 'check':
        check()
    elif c == 'run':
        run(sys.argv[2], sys.argv[3])
    elif c == 'fetch':
        fetch(sys.argv[2], '--wait' in sys.argv)
