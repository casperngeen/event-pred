"""Live path of the exchange simulator: tailing a growing file, the clock lag, restart handling, the shadow book."""
import json
import tempfile
import time
import unittest
from pathlib import Path
import live_exec as LE
from test_event_execution import TK, snapshot, delta, print_


def line(m, sent_ms):
    return json.dumps(dict(received_at='2026-10-30T00:00:00.000+00:00', sending_ts_ms=sent_ms, **m)) + '\n'


class Shadow:
    def __init__(self):self.placed, self.cancelled, self.updated, self.busy = [], [], 0, False
    def is_busy(self, tk):return self.busy
    def place(self, d):self.placed.append(d)
    def cancel_entries(self, t):self.cancelled.append(t)
    def update(self, mk, now):self.updated += 1


class TestLiveExec(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.x = LE.LiveExecution(lambda n, r: self.events.append(r), None, {TK}, 0., safety=1.)

    def test_book_is_advanced_only_to_wall_time_minus_safety(self):
        self.x.feed(100.0, snapshot([['.40', '10']], [['.40', '10']]))
        self.x.feed(100.6, delta('yes', '.40', 5))
        self.x.update(None, 100.5);self.assertNotIn(TK, self.x.levels)      # snapshot at 100.0 is within safety of 100.5
        self.x.update(None, 101.2);self.assertEqual(self.x.levels[TK]['yes'][4000], 10.)
        self.x.update(None, 101.7);self.assertEqual(self.x.levels[TK]['yes'][4000], 15.)

    def test_order_sent_at_wall_time_fills_on_prints_after_it(self):
        self.x.latency = .1
        self.x.feed(100.0, snapshot([['.40', '10']], [['.40', '10']]));self.x.update(None, 102.)
        self.x.feed(102.05, print_(49, 5));self.x.feed(102.8, print_(49, 5))
        self.x.update(None, 102.5)                                          # wall 102.5, book at 101.5
        self.x.place(dict(t=102., ticker=TK, kind='spread', side=1, limit=50, size=5, signal=10))
        self.x.update(None, 104.)
        self.assertEqual(self.x.lots[0]['t_fill'], 102.8)                   # the 102.05 print precedes acceptance at 102.5+.1

    def test_late_message_is_applied_and_counted(self):
        self.x.feed(100.0, snapshot([['.40', '10']], [['.40', '10']]));self.x.update(None, 105.)
        self.x.feed(101.0, delta('yes', '.40', 3));self.x.update(None, 106.)
        self.assertEqual(self.x.stats['late_messages'], 1);self.assertEqual(self.x.levels[TK]['yes'][4000], 13.)

    def test_catch_up_does_not_move_the_clock_past_the_backlog(self):
        old = LE.BATCH;LE.BATCH = 2
        try:
            self.x.feed(100.0, snapshot([['.40', '10']], [['.40', '10']]))
            for i in range(1, 5):self.x.feed(100. + i, delta('yes', '.40', 1))
            self.x.update(None, 1000.)
            self.assertLess(self.x.now, 105.);self.assertEqual(self.x.stats['late_messages'], 0)
            for _ in range(5):self.x.update(None, 1000.)
            self.assertEqual(self.x.levels[TK]['yes'][4000], 14.);self.assertEqual(self.x.stats['late_messages'], 0)
        finally:LE.BATCH = old

    def test_shadow_gets_the_same_order_unless_busy(self):
        sh = Shadow();self.x.shadow = sh
        self.x.feed(100.0, snapshot([['.40', '10']], [['.40', '10']]));self.x.update(None, 102.)
        d = dict(t=102., ticker=TK, kind='spread', side=1, limit=50, size=5, signal=10)
        self.x.place(d);self.assertEqual(sh.placed, [d])
        self.x.update(object(), 103.);self.assertEqual(sh.updated, 1)
        sh.busy = True;self.x.busy[TK] = None;self.x.place(dict(d, t=103.));self.assertEqual(len(sh.placed), 1);self.assertEqual(self.x.stats['shadow_skipped_busy'], 1)

    def test_restart_continues_ids_and_logs_abandoned_orders(self):
        prior = [dict(event='submitted', id=0), dict(event='submitted', id=1), dict(event='entry_cancel_ack', id=0), dict(event='fill', id=0)]
        self.assertEqual(LE.LiveExecution.resume_point(prior), 2)
        self.x.id0 = 2;self.x.resume(prior)
        a = [e for e in self.events if e['event'] == 'abandoned_on_restart'][0]
        self.assertEqual(a['resting_entries'], [1]);self.assertEqual(a['earlier_fills'], 1)
        self.x.feed(100.0, snapshot([['.40', '10']], [['.40', '10']]));self.x.update(None, 102.)
        self.x.place(dict(t=102., ticker=TK, kind='spread', side=1, limit=50, size=1, signal=10));self.assertEqual(self.x.orders[0]['id'], 2)
        self.assertFalse(self.x.is_busy('other'))

    def test_tail_follows_a_growing_file_and_waits_for_complete_lines(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'books_ws.jsonl'
            got = [];path.touch()
            tail = LE.BookTail(path, {TK}, lambda ts, m: got.append((ts, m['type'])), poll=.02);tail.start()
            with path.open('a') as f:
                f.write(line(snapshot([['.40', '10']], [['.40', '10']]), 100000));f.flush()
                half = line(delta('yes', '.40', 1), 100500);f.write(half[:30]);f.flush();time.sleep(.2)
                self.assertEqual(len(got), 1)                                # the half line is not parsed
                f.write(half[30:]);f.write(line(dict(type='orderbook_delta', msg=dict(market_ticker='OTHER', side='yes', price_dollars='.40', delta_fp='1')), 100600));f.flush()
            time.sleep(.3);tail.stop_event.set()
            self.assertEqual([t for _, t in got], ['orderbook_snapshot', 'orderbook_delta'])
            self.assertAlmostEqual(got[0][0], 100.0 - .004, places=6)


if __name__ == '__main__':unittest.main(verbosity=2)
