"""Execution-realism sensitivity checks, separate from b38 strategy selection.

The current paper engine waits through the grace interval before posting an
exit but backdates its price/fill search to fill+hold. Compare causally posted
exits and require enough through-print volume for full limit fills.
"""
from __future__ import annotations

import argparse
import ast
import concurrent.futures
import hashlib
import inspect
import json
import multiprocessing
import textwrap
import time
from pathlib import Path

import b38_strategy_study as S

OUT=Path(__file__).with_name('out')/'b39_execution_checks'
CANDIDATES=[dict(name='baseline'),dict(name='causal_actual_post',causal='actual_post'),
            dict(name='causal_hold_deadline',causal='hold'),dict(name='volume_confirmed_limits',volume=True),
            dict(name='causal_and_volume',causal='hold',volume=True)]


class PhysicalBook(S.Book):
    def __init__(self,cfg):
        super().__init__(cfg)
        self.volume=cfg.get('volume',False);self.need=1.0
        if cfg.get('causal'):
            tree=ast.parse(textwrap.dedent(inspect.getsource(S.B.T.PaperBook.update)))
            changed=0
            for node in ast.walk(tree):
                if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='te':
                    node.value=ast.Name(id='now',ctx=ast.Load());changed+=1
                if cfg['causal']=='hold' and isinstance(node,ast.If) and ast.unparse(node.test).startswith("o['stage'] == 'hold' and"):
                    node.test=ast.parse("o['stage']=='hold' and now>=o['t_fill']+HOLD_S",mode='eval').body
            assert changed==1,changed
            env=dict(self.raw_update.__globals__)
            exec(compile(ast.fix_missing_locations(tree),'<causal exit timing>','exec'),env)
            self.raw_update=env['update']

    def _fill(self,p,a,b,rest,price,ahead):
        if not self.volume:
            return super()._fill(p,a,b,rest,price,ahead)
        np=S.B.LM.np
        i,j=np.searchsorted(p['t'],a,side='right'),np.searchsorted(p['t'],b,side='right')
        side=p['yes'][i:j] if rest==-1 else ~p['yes'][i:j]
        px=p['px'][i:j]
        hit=side & ((px>price) if rest==-1 else (px<price))
        enough=np.cumsum(np.where(hit,p['n'][i:j],0.0))>=self.need-1e-9
        return i+int(np.argmax(enough)) if enough.any() else -1

    def update(self,mk,now):
        history=self.orders
        active=sorted((o for o in self.active() if o['stage'] not in ('closed','expired','settle')),key=lambda o:o['id'])
        try:
            for o in active:
                self.orders=[o];self.need=o['size']
                self.raw_update(self,mk,now)
        finally:
            self.orders=history
        self.peak_risk=max(self.peak_risk,self.risk())


def run_one(p,study_hash):
    S.OUT=OUT;S.CANDIDATES=CANDIDATES;S.Book=PhysicalBook
    return S.run(Path(p),study_hash)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=2);ap.add_argument('--source-prefix');ap.add_argument('--game',action='append');a=ap.parse_args()
    S.OUT=OUT;S.CANDIDATES=CANDIDATES
    rule,_=S.fingerprint()
    rule.update(execution_source_sha256=S.B.digest(Path(__file__)),
                interpretation='sensitivity checks, not a complete queue/impact model; taker-print fallback is unchanged')
    h=hashlib.sha256(json.dumps(rule,sort_keys=True).encode()).hexdigest()
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'study_rule.json').write_text(json.dumps(rule,indent=2)+'\n')
    files=[]
    for p in sorted(S.B.OUT.glob('recorded/*/*/result.json')):
        r=json.loads(p.read_text())
        if r['phase']=='preseason' or r.get('manifest_hash')!=rule['base_rule_sha256']:continue
        if a.source_prefix and not r['root'].startswith(a.source_prefix):continue
        if a.game and r['game'] not in a.game:continue
        files.append(p)
    print(json.dumps(dict(games=len(files),study_hash=h)),flush=True)
    results=[];errors=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        tasks={pool.submit(run_one,str(p),h):p for p in files}
        for task in concurrent.futures.as_completed(tasks):
            try:
                r=task.result();results.append(r)
                print(json.dumps(dict(done=len(results),game=r['game'],baseline=r.get('results',{}).get('baseline',{}).get('pnl_c'),excluded=r.get('excluded'))),flush=True)
            except Exception as ex:
                import traceback
                e=dict(game=tasks[task].parent.name,error=repr(ex),traceback=traceback.format_exc());errors.append(e);print(json.dumps(e),flush=True)
    (OUT/'errors.json').write_text(json.dumps(errors,indent=2)+'\n')


if __name__=='__main__':main()
