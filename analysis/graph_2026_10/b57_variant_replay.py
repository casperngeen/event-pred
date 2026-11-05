"""b49 event replay with one engine safeguard switched off, for the guard / late-block comparisons.

  VARIANT=noguard   every feed-lateness check off: MLB ball-in-play wait; NBA/NFL last-5-plays-late
                    stall check; NBA review / placeholder wait and the real-time-stamp gate
  VARIANT=nolate    NBA late-game totals block off (everything else as in b49)

Run exactly like b49 (OUT_NAME, GAMES, ARMS, SIZING, WORKERS env; --monitor --local-only). b49's source is untouched, so its
rule_hash and cached results stay valid. The patches sit at module level because b49 workers re-import the main module.
"""
import os
import sys

import b49_event_replay_v2 as BB
import b64_fast_stream  # noqa: F401  (orjson loader, identical stream)
import b74_pin_inputs  # noqa: F401  (same recorded inputs as the main run)

VARIANT = os.environ['VARIANT']
assert VARIANT in ('noguard', 'nolate', 'noguard_nolate'), VARIANT
T = BB.B.T


class Off:
    """Data descriptor: reads False, ignores assignment (b49 and Engine.__init__ both assign these attributes)."""
    def __get__(self, obj, cls=None):
        return False

    def __set__(self, obj, value):
        pass


if VARIANT in ('noguard', 'noguard_nolate'):
    T.Engine.gate_true = Off()
    T.Engine.skip_in_play = Off()
    T.Engine._stale = lambda self: None
    T.Engine._unsettled = lambda self, t: None
if VARIANT in ('nolate', 'noguard_nolate'):
    T.Engine.late_block = Off()

if __name__ == '__main__':
    sys.exit(BB.main())
