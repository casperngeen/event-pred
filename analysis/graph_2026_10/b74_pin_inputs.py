"""Pin a wrapper replay (b57 variants, b60 baselines) to the recorded inputs the main run used for each game.

Every game is recorded on two hosts (VM and sandbox) with slightly different arrival stamps, and b49's inventory picks, per game, the
root with the most feed files, which can change between runs as folders gain files. A variant replayed from the other copy differs
from the main run for reasons that have nothing to do with the switch under test (seen on 27 and 29 Oct NBA games, 2026-11-01).
Import this module (after b49) before running a wrapper: for every (date, game) that b55_main_20x's result.json records, the
entry's root and book are replaced by the main run's. Games the main run does not have keep the inventory's choice.
Not imported by b49 itself, so b49's rule_hash and its cached results are unchanged.
"""
import json
from pathlib import Path

import b49_event_replay_v2 as BB

MAIN = Path(__file__).with_name('out') / 'b55_main_20x' / 'games'
_inventory = BB.inventory
_pins = None


def pins():
    global _pins
    if _pins is None:
        _pins = {}
        for f in MAIN.glob('*/*/result.json'):
            r = json.loads(f.read_text())
            if r.get('root') and r.get('book'):
                _pins[(r['date'], r['league'], str(r['game']))] = (r['root'], r['book'])
    return _pins


def inventory(fetch=True):
    entries, errors = _inventory(fetch)
    p = pins()
    for e in entries:
        k = (e['date'], e['league'], str(e['game']))
        if k in p and (BB.B.ROOT / p[k][0] / 'game.json').exists() and (BB.B.ROOT / p[k][1]).exists():
            e['root'], e['book'] = p[k]
    return entries, errors


BB.inventory = inventory
