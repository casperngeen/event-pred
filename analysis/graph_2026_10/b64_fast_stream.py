"""Faster drop-in for event_execution.load_book_stream: orjson instead of json (the parse was a third of the load time).

Same filtering, same EventStream.add / finalize calls, so the stream is identical; test_fast_stream() checks that on one book.
Import this module before running b49 from a wrapper (b57, b60): it patches X.load_book_stream. It is not imported by b49
itself, so b49's source and its rule_hash (and every cached result) are unchanged.
"""
import gzip
import math
from datetime import datetime
import orjson

import b49_event_replay_v2 as BB

X = BB.X
_orig = X.load_book_stream


def load_book_stream(path, tickers, diagnostics=None):
    op = gzip.open if str(path).endswith('.gz') else open
    st = X.EventStream()
    kinds = ('orderbook_snapshot', 'orderbook_delta', 'trade')
    ch = X.Chains()
    loads, add, chain = orjson.loads, st.add, ch.chain
    try:
        with op(path, 'rb') as f:
            for line in f:
                try:
                    m = loads(line)
                except ValueError:
                    if diagnostics is not None:
                        diagnostics['bad_json_lines'] += 1
                    continue
                t = m.get('type')
                if t in kinds and (m.get('msg') or {}).get('market_ticker') in tickers:
                    add(m, chain=chain(m.get('sid'), m.get('seq')) if t != 'trade' else 0)
                    if t == 'orderbook_delta' and len(st) % 500 == 0 and m.get('received_at'):
                        st.lag_sample.append(datetime.fromisoformat(m['received_at'].replace('Z', '+00:00')).timestamp() - st.ts[-1])
    except (EOFError, OSError) as ex:
        if diagnostics is not None:
            diagnostics['truncated_stream'] = repr(ex)
    xs = sorted(st.lag_sample)
    st.lag_median = xs[len(xs) // 2] if xs else math.inf
    return st.finalize()


def test_fast_stream(path, tickers):
    a, b = _orig(path, tickers), load_book_stream(path, tickers)
    same = all(getattr(a, n) == getattr(b, n) for n in ('ts', 'kind', 'tk', 'side', 'pk', 'val', 'chain', 'raw', 'names')) \
        and a.payload == b.payload and a.lag_median == b.lag_median
    return same, len(a)


X.load_book_stream = load_book_stream

import b74_pin_inputs  # noqa: E402,F401  (every wrapper that loads this also replays the main run's recorded inputs)
