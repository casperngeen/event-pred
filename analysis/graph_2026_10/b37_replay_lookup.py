"""Exact, bounded version of the current engine's recent-side print lookup.

Patch only the two list-comprehension assignments in Engine._decide. No signal,
guard, sizing, order or exit logic changes; production source stays untouched.
"""
import ast
import inspect
import textwrap

import numpy as np


def recent_side(p, n, t, yes, width):
    # The small padding makes the candidate slice conservative even at floating
    # point boundaries. Apply the original subtraction comparison exactly.
    lo = int(np.searchsorted(p['t'], t-width-.001, side='left'))
    side = p['yes'][lo:n] if yes else ~p['yes'][lo:n]
    hit = np.flatnonzero(side & (t-p['t'][lo:n] <= width))
    return [lo+int(hit[-1])] if len(hit) else []


def install(engine):
    original = engine._decide
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    changed = 0
    for node in ast.walk(tree):
        if not isinstance(node,ast.Assign) or len(node.targets)!=1 or not isinstance(node.targets[0],ast.Name):
            continue
        name = node.targets[0].id
        if name not in ('yes_i','no_i'):
            continue
        assert isinstance(node.value,ast.Subscript) and isinstance(node.value.value,ast.ListComp), ast.dump(node)
        node.value = ast.Call(func=ast.Name(id='_replay_recent_side',ctx=ast.Load()),
                              args=[ast.Name(id='p',ctx=ast.Load()),ast.Name(id='n',ctx=ast.Load()),
                                    ast.Name(id='t',ctx=ast.Load()),ast.Constant(value=name=='yes_i'),
                                    ast.Name(id='WIDTH_S',ctx=ast.Load())],keywords=[])
        changed += 1
    assert changed==2,changed
    env = dict(original.__globals__,_replay_recent_side=recent_side)
    exec(compile(ast.fix_missing_locations(tree),'<bounded replay print lookup>','exec'),env)
    engine._decide = env['_decide']
