"""Execution invariants, including the previously missed 72c maker exit."""
import unittest
import json
from decimal import Decimal as D
from event_execution import EventStream,Execution,FeeLedger,model_fee,size_contracts,snap

TK='KXNBASPREAD-T'


def snapshot(yes,no,tk=TK):
    return dict(type='orderbook_snapshot',msg=dict(market_ticker=tk,yes_dollars_fp=yes,no_dollars_fp=no))


def delta(side,price,change,tk=TK):
    return dict(type='orderbook_delta',msg=dict(market_ticker=tk,side=side,price_dollars=price,delta_fp=str(change)))


def print_(price,size,side='no',tid=None,ts=None,tk=TK):
    d=dict(market_ticker=tk,yes_price_dollars=str(price/100),count_fp=str(size),taker_side=side)
    if tid:d['trade_id']=tid
    if ts:d['ts_ms']=ts*1000
    return dict(type='trade',msg=d)


class TestFees(unittest.TestCase):
    def test_kalshi_doc_example(self):
        # docs.kalshi.com fee_rounding: revenue -$0.055, model fee $0.00363825 -> trade 0.003639 + rounding 0.001361 = $0.005
        self.assertEqual(FeeLedger(D('0.01')).charge(D('-0.055'),D('0.00363825')),D('0.005000'))
        self.assertEqual(FeeLedger().charge(D('-0.055'),D('0.00363825')),D('0.003700'))   # direct member: aligned to $0.0001

    def test_accumulator_rebates_across_fills(self):
        led=FeeLedger(D('0.01'));fees=[led.charge(D('-0.055'),D('0.00363825')) for _ in range(3)]
        self.assertTrue(all(f>=0 for f in fees));self.assertEqual(sum(fees),D('0.015'))

    def test_series_rules(self):
        self.assertEqual(model_fee('KXMLBTOTAL-X',True,50,1),0)             # taker-only series: no maker fee
        self.assertEqual(model_fee('KXMLBTOTAL-X',False,50,1),D('0.07')*D('0.5')*D('0.25'))
        self.assertEqual(model_fee('KXNBATOTAL-X',True,50,2),D('0.0175')*2*D('0.25'))
        self.assertEqual(model_fee('KXMLBGAME-X',False,50,1),D('0.07')*D('0.5')*D('0.25'))

    def test_rounds_up_to_cent_per_order(self):
        # 0.5 contract taker at 50c: model fee 0.00875 but balance precision is 1c, so the order pays 1c.
        self.assertEqual(FeeLedger(D('0.01')).charge(D('-0.25'),model_fee(TK,False,50,.5)),D('0.01'))


class TestExecution(unittest.TestCase):
    def test_user_sizing(self):
        self.assertEqual(size_contracts(.321),1.61)
        self.assertEqual(size_contracts(1.),5.)
        self.assertEqual(size_contracts(.001),.01)
        self.assertEqual(size_contracts(0.),0.)

    def test_snap_to_cent_grid(self):
        self.assertEqual(snap(52.7,1),52.);self.assertEqual(snap(52.2,-1),53.);self.assertEqual(snap(52.,1),52.)
        self.assertEqual(snap(0.4,1),1.);self.assertEqual(snap(99.6,-1),99.)

    def setUp(self):
        self.events=[]
        self.x=Execution(lambda name,r:self.events.append(r),0.)
        self.x.advance(100)
        self.x.message(snapshot([['.40','10']],[['.40','10']]))

    def place(self,side=1,limit=50,size=5):
        self.x.place(dict(t=100,ticker=TK,kind='spread',side=side,limit=limit,size=size,signal=10))
        self.x.advance(self.x.now)
        return self.x.orders[-1]

    def trade(self,*a,**k):self.x.message(print_(*a,**k))

    def test_limit_snapped_on_place(self):
        p=self.place(limit=49.6);self.assertEqual(p['limit'],49.)

    def test_through_trade_fills_in_full_regardless_of_print_size(self):
        p=self.place();self.x.advance(101);self.trade(49,.01)
        self.assertEqual(p['pending'],0);self.assertEqual(sum(x['size'] for x in self.x.lots),5)
        self.assertEqual(self.x.lots[0]['entry'],50.)

    def test_volume_mode_limits_through_fill_to_print_size(self):
        self.x.through='volume';p=self.place();self.x.advance(101);self.trade(49,2)
        self.assertEqual(sum(l['size'] for l in self.x.lots),2);self.assertEqual(p['pending'],3)

    def test_strict_repair_keeps_ticker_invalid_until_snapshot(self):
        self.x.repair='strict';self.x.message(delta('yes','.40',-11));self.x.advance(self.x.now+60)
        self.assertFalse(self.x.healthy(TK));self.x.message(snapshot([['.40','3']],[['.40','4']]));self.assertTrue(self.x.healthy(TK))

    def test_at_limit_queue(self):
        p=self.place(limit=40);self.x.advance(101);self.trade(40,10)
        self.assertFalse(self.x.lots)
        self.trade(40,2);self.assertEqual(self.x.lots[0]['size'],2)

    def test_wrong_side_and_at_higher_price_do_not_fill(self):
        self.place();self.trade(49,100,'yes');self.trade(51,100,'no')
        self.assertFalse(self.x.lots)

    def test_snapshot_replaces_book(self):
        self.x.message(snapshot([['.30','4']],[['.20','8']]))
        self.assertEqual(self.x.levels[TK]['yes'],{3000:4.})

    def test_crossing_walks_multiple_levels(self):
        self.x.message(snapshot([['.40','10']],[['.50','2'],['.49','3']]))
        p=self.place(limit=51,size=5)
        self.assertEqual([(l['entry'],l['size']) for l in self.x.lots],[(50.,2.),(51.,3.)])
        self.assertEqual(p['pending'],0)
        self.assertTrue(all(not l['entry_maker'] for l in self.x.lots))

    def test_missing_book_rejects(self):
        self.x.valid[TK]=False;p=self.place()
        self.assertTrue(p['entry_done']);self.assertFalse(self.x.is_busy(TK));self.assertFalse(self.x.lots)

    def test_exchange_time_ordering_ignores_receive_order(self):
        # Stream is exchange-time ordered: a print at 100.05 precedes acceptance at 100.1, so it cannot fill us;
        # the print at 100.2 can.
        x=Execution(lambda n,r:None,.1);x.advance(99.9)
        x.attach(EventStream.from_messages([(99.95,snapshot([['.40','10']],[['.40','10']])),(100.2,print_(49,5)),(100.05,print_(49,5))]))
        x.advance(100)
        x.place(dict(t=100,ticker=TK,kind='spread',side=1,limit=50,size=5,signal=10))
        x.advance(100.15);self.assertFalse(x.lots)
        x.advance(100.25);self.assertEqual(x.lots[0]['t_fill'],100.2)

    def test_stream_sorts_and_applies_deltas(self):
        x=Execution(lambda n,r:None,0.)
        x.attach(EventStream.from_messages([(2.,delta('yes','.40',3)),(1.,snapshot([['.40','10']],[['.40','10']])),(3.,delta('yes','.40',-13))]))
        x.advance(2.5);self.assertEqual(x.levels[TK]['yes'][4000],13.)
        x.advance(3.5);self.assertNotIn(4000,x.levels[TK]['yes'])

    def test_cancellation_does_not_advance_queue(self):
        self.place(limit=40)
        self.x.message(delta('yes','.40',-10))
        self.x.advance(101);self.trade(40,2)
        self.assertFalse(self.x.lots)

    def test_entry_expires_and_cannot_fill_later(self):
        p=self.place();self.x.advance(161);self.trade(49,10)
        self.assertTrue(p['entry_done']);self.assertFalse(self.x.lots)

    def test_latency(self):
        self.x.latency=.1;p=self.place();self.x.advance(100.05);self.trade(49,5)
        self.assertFalse(self.x.lots);self.x.advance(100.1);self.x.advance(100.2);self.trade(49,5)
        self.assertEqual(self.x.lots[0]['t_fill'],100.2)

    def test_duplicate_trade_does_not_fill_twice(self):
        self.place(limit=40);self.x.advance(101)
        self.trade(40,12,tid='same');self.trade(40,12,tid='same')
        self.assertEqual(sum(l['size'] for l in self.x.lots),2)

    def test_72c_exit_fills_from_76c_buy_with_bid_still_70(self):
        self.place(limit=50);self.x.advance(101);self.trade(49,5)
        self.x.ref=lambda tk,t:71.
        self.x.message(snapshot([['.70','20']],[['.24','100']]))
        self.x.advance(221)
        self.trade(76,97.4,side='yes')
        lot=self.x.lots[0]
        self.assertEqual(lot['remaining'],0)
        self.assertEqual(lot['exits'][0]['price'],72.)
        self.assertEqual(max(self.x.levels[TK]['yes']),7000)

    def test_settlement_accounting(self):
        self.place();self.x.advance(101);self.trade(50,2)
        a=self.x.results(lambda tk:100)['by_kind']['spread']
        # 2 contracts bought at 50 as maker: model fee 0.00875 (direct), the order total rounds up to 1c on a $0.01 balance.
        self.assertAlmostEqual(a['fees_c'],.88);self.assertAlmostEqual(a['pnl_c'],100-.88)
        self.assertAlmostEqual(a['fees_1c_c'],1.0);self.assertAlmostEqual(a['pnl_1c_c'],99.0)
        self.assertEqual(a['settled_contracts'],2)

    def test_trade_volume_shared_across_multiple_exit_lots(self):
        self.place();self.x.advance(101);self.trade(50,2)
        self.x.advance(102);self.trade(50,3)
        self.x.ref=lambda tk,t:71.
        self.x.message(snapshot([['.70','20']],[['.24','100']]))
        self.x.advance(222)
        self.trade(72,3,side='yes')
        self.assertEqual(sum(x['size'] for l in self.x.lots for x in l['exits']),3)
        self.assertEqual(sum(l['remaining'] for l in self.x.lots),2)

    def test_unknown_delta_invalidates_until_snapshot(self):
        self.x.message(delta('yes','.40',-11))
        self.assertFalse(self.x.healthy(TK))
        self.x.message(snapshot([['.40','3']],[['.40','4']]))
        self.assertTrue(self.x.healthy(TK))

    def test_taken_liquidity_clamps_to_real_level(self):
        self.x.message(snapshot([['.40','10']],[['.50','5']]))
        self.place(limit=50,size=3)                                   # takes 3 of the 5 at 50
        self.assertEqual(self.x.used[(TK,'no',5000)],3.)
        self.x.message(delta('no','.50',-4))                          # real level 5 -> 1: we cannot hold 3
        self.assertEqual(self.x.used[(TK,'no',5000)],1.)
        self.x.message(delta('no','.50',9))                           # real level back to 10
        self.assertEqual(self.x.levels[TK]['no'][5000]-self.x.used[(TK,'no',5000)],9.)

    def test_lots_export_has_no_internal_order_cycles(self):
        self.place();self.x.advance(101);self.trade(50,2)
        self.x.ref=lambda tk,t:71.
        self.x.message(snapshot([['.70','20']],[['.24','100']]))
        self.x.advance(221)
        exported=json.loads(json.dumps(self.x.export_lots()))
        self.assertEqual(exported[0]['size'],2)
        self.assertNotIn('exit_order',exported[0])


if __name__=='__main__':unittest.main(verbosity=2)
