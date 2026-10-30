"""Event-driven L2 execution simulator for Kalshi paper orders. No network or live-order submission.

Clock: exchange time. Book messages are replayed in the order Kalshi produced them (delta / trade `ts_ms`, snapshot
`sending_ts_ms`), not the order we received them, so an order accepted at exchange time T sees exactly the book that existed
at T. Our decision time is the (NTP-synced) local clock; `latency` is decision -> exchange acceptance.

Queue: new resting orders join the back of their price level; trades, not deltas, consume the queue (cancellations never
advance us). A print strictly through our limit means the level was exhausted, so we are filled in full without drawing on
that print's volume. Prices are snapped to the whole-cent grid (valid on every Kalshi price structure).

Liquidity we take is subtracted from the real level and stays subtracted, clamped to the real level whenever it shrinks
(the counterfactual level cannot be negative). Historical book updates remain exogenous (no market impact).

Fees: per-series type and multiplier (GET /series), taker 0.07 and maker 0.0175 of p(1-p) per contract, with Kalshi's
per-order rounding fee and rebate accumulator (docs.kalshi.com/getting_started/fee_rounding.md). Balance precision is
$0.0001 for direct members (headline) and $0.01 for FCM-cleared members; both are accounted on every fill (`fee_c`, `fee1c_c`).
"""
from __future__ import annotations
import array
import bisect
import collections
import gzip
import heapq
import json
import math
from datetime import datetime
from decimal import Decimal as D, ROUND_CEILING, ROUND_FLOOR

EPS=1e-8
SUSPECT_S=5.        # after a book inconsistency (negative level, crossed book) the ticker is not traded for this long
MICRO=D('0.000001');DIRECT=D('0.0001');CENT=D('0.01')
RATES=dict(taker=D('0.07'),maker=D('0.0175'))
# fee_type, fee_multiplier from GET /series/{ticker}, unchanged since 2026-08-07 (and no event overrides), checked 2026-10-30.
SERIES_FEES={'KXMLBGAME':('quadratic_with_maker_fees',D('0.5')),'KXMLBSPREAD':('quadratic',D('0.5')),'KXMLBTOTAL':('quadratic',D('0.5')),
             'KXNBAGAME':('quadratic_with_maker_fees',D(1)),'KXNBASPREAD':('quadratic_with_maker_fees',D(1)),'KXNBATOTAL':('quadratic_with_maker_fees',D(1)),
             'KXNFLGAME':('quadratic_with_maker_fees',D(1)),'KXNFLSPREAD':('quadratic_with_maker_fees',D(1)),'KXNFLTOTAL':('quadratic_with_maker_fees',D(1))}


def size_contracts(weight):
    return math.floor(500*weight+.5)/100


def snap(price_c,side):
    """Whole-cent limit: a buy rounds down, a sell rounds up (never more aggressive than intended), kept inside 1-99c."""
    px=math.floor(price_c+EPS) if side==1 else math.ceil(price_c-EPS)
    return float(min(99,max(1,px)))


def model_fee(ticker,maker,price_c,qty):
    """Exchange fee-model dollars for one fill, before rounding."""
    kind,mult=SERIES_FEES[ticker.split('-')[0]]
    if maker and kind!='quadratic_with_maker_fees':return D(0)
    p=D(repr(round(price_c,6)))/100
    return mult*RATES['maker' if maker else 'taker']*D(repr(round(qty,2)))*p*(1-p)


class FeeLedger:
    """One per order: the fee accumulator carries rounding overpayment across that order's fills."""
    def __init__(self,precision=DIRECT):self.acc=D(0);self.prec=precision

    def charge(self,revenue,fee):
        """revenue: signed dollars (negative when buying yes). Returns the net fee in dollars."""
        trade=fee.quantize(MICRO,ROUND_CEILING);change=revenue-trade
        rounding=change-(change/self.prec).to_integral_value(ROUND_FLOOR)*self.prec
        self.acc+=rounding
        rebate=min((self.acc/self.prec).to_integral_value(ROUND_FLOOR)*self.prec,trade+rounding)
        self.acc-=rebate
        return trade+rounding-rebate


SEND_DELAY=.004     # sending_ts_ms - ts_ms, median 4 ms: one consistent exchange clock for deltas, trades and snapshots


def book_ts(m):
    """Exchange-time seconds for a raw WebSocket message. One clock for every message type: mixing delta `ts_ms` with
    snapshot `sending_ts_ms` reorders a snapshot against the deltas around it and corrupts the book."""
    if m.get('sending_ts_ms') is not None:return float(m['sending_ts_ms'])/1000-SEND_DELAY
    d=m.get('msg') or {}
    if d.get('ts_ms') is not None:return float(d['ts_ms'])/1000
    return datetime.fromisoformat(m['received_at'].replace('Z','+00:00')).timestamp()-.014


class EventStream:
    """Exchange-time book messages as compact arrays (a game is millions of deltas; dicts do not fit in memory)."""
    DELTA,TRADE,SNAPSHOT=0,1,2

    def __init__(self):
        self.ts=array.array('d');self.kind=array.array('b');self.tk=array.array('i');self.side=array.array('b')
        self.pk=array.array('i');self.val=array.array('d');self.chain=array.array('i');self.raw=array.array('d');self.payload=[];self.names=[];self.ids={}
        self.lag_sample=[];self.dropped=0;self.chains=0;self.lifted=0

    def __len__(self):return len(self.ts)

    def add(self,m,ts=None,chain=0):
        d=m.get('msg') or {};tk=d.get('market_ticker');kind=m.get('type')
        if kind not in ('orderbook_snapshot','orderbook_delta','trade') or not tk:return
        i=self.ids.get(tk)
        if i is None:i=self.ids[tk]=len(self.names);self.names.append(tk)
        self.ts.append(book_ts(m) if ts is None else ts);self.tk.append(i);self.chain.append(chain)
        self.raw.append(float(d['ts_ms']) if d.get('ts_ms') is not None else 0.)
        if kind=='orderbook_delta':
            self.kind.append(0);self.side.append(0 if d['side']=='yes' else 1)
            self.pk.append(round(float(d['price_dollars'])*10000) if d.get('price_dollars') is not None else round(float(d['price'])*100))
            self.val.append(float(d.get('delta_fp',d.get('delta',0))))
        else:
            self.kind.append(1 if kind=='trade' else 2);self.side.append(0);self.pk.append(len(self.payload));self.val.append(0.)
            self.payload.append(m)

    def finalize(self):
        """Merge the recorded connections into one change log.
        1. Every connection carries true exchange changes (one can go quiet while another delivers), so events are merged,
           not one connection chosen: an event seen on several connections is kept once, keyed on (ticker, ts_ms, side,
           price, delta) plus its occurrence number within its own connection, so genuinely repeated same-millisecond
           changes (a market maker pulling several equal orders) are all kept.
        2. A snapshot is stamped when sent, not when its state was read, so within its own connection and ticker it is
           clipped to lie between the deltas recorded before it and those recorded after it.
        3. Sort by exchange time."""
        n=len(self.ts);groups=collections.defaultdict(list)
        for i in range(n):
            if self.chain[i]:groups[self.chain[i],self.tk[i]].append(i)
        for idx in groups.values():
            if not any(self.kind[i]==EventStream.SNAPSHOT for i in idx):continue
            lo=-math.inf;lows={}
            for i in idx:
                if self.kind[i]==EventStream.SNAPSHOT:lows[i]=lo
                else:lo=max(lo,self.ts[i])
            hi=math.inf
            for i in reversed(idx):
                if self.kind[i]==EventStream.SNAPSHOT:
                    new=min(max(self.ts[i],lows[i]),hi)
                    if new!=self.ts[i]:self.ts[i]=new;self.lifted+=1
                    hi=math.inf                    # deltas before this snapshot belong to the previous segment
                else:hi=min(hi,self.ts[i])
        self.chains=len({c for c in self.chain if c})
        if self.chains>1:
            keep=array.array('b',[1])*n;seen=set();cnt=collections.Counter();order=collections.deque()
            for i in range(n):
                if self.kind[i]!=EventStream.DELTA:continue
                key=(self.tk[i],self.raw[i],self.side[i],self.pk[i],self.val[i]);c=(self.chain[i],key);occ=cnt[c];cnt[c]=occ+1;ev=(key,occ)
                if ev in seen:keep[i]=0;self.dropped+=1;continue
                seen.add(ev);order.append((self.raw[i],c,ev))
                if len(order)>400_000:
                    horizon=self.raw[i]-120_000
                    while order and order[0][0]<horizon:
                        _,cc,e=order.popleft();seen.discard(e);cnt.pop(cc,None)
            if self.dropped:
                idx=[i for i in range(n) if keep[i]]
                for name,code in (('ts','d'),('kind','b'),('tk','i'),('side','b'),('pk','i'),('val','d'),('chain','i'),('raw','d')):
                    old=getattr(self,name);setattr(self,name,array.array(code,(old[i] for i in idx)))
        return self.sort()

    def sort(self):
        n=len(self.ts);ts=self.ts
        if any(ts[i]>ts[i+1] for i in range(n-1)):
            order=sorted(range(n),key=ts.__getitem__)           # stable: ties keep recorded order
            for name,code in (('ts','d'),('kind','b'),('tk','i'),('side','b'),('pk','i'),('val','d'),('chain','i'),('raw','d')):
                old=getattr(self,name);setattr(self,name,array.array(code,(old[i] for i in order)))
        return self

    @classmethod
    def from_messages(cls,pairs):
        """pairs: [(exchange_ts, message)] (tests)."""
        st=cls()
        for ts,m in pairs:st.add(m,ts)
        return st.sort()


class Chains:
    """The recorder file can hold several WebSocket connections at once (a duplicated runner, a resubscribe while the old
    connection is still delivering): each carries every change, so applying all of them to one book double-counts. A
    connection's `seq` only increases, so a message belongs to the chain whose last seq is the greatest one below it."""
    GAP=200_000

    def __init__(self):self.tails={};self.cid={};self.n=0

    def chain(self,sid,seq):
        """Chain id (>0) of an order-book message; 0 for anything without a sequence number."""
        if seq is None:return 0
        tails=self.tails.setdefault(sid,[]);cid=self.cid.setdefault(sid,[]);i=bisect.bisect_left(tails,seq)
        if i and seq-tails[i-1]<=self.GAP:
            tails[i-1]=seq;return cid[i-1]
        tails.insert(i,seq);self.n+=1;cid.insert(i,self.n);return self.n


def load_book_stream(path,tickers,diagnostics=None):
    """Raw books_ws file -> EventStream for `tickers`, in exchange-time order, one connection feeding the book at a time
    (see EventStream.finalize); `.lag_median` = median receive lag of deltas."""
    op=gzip.open if str(path).endswith('.gz') else open;st=EventStream();kinds=('orderbook_snapshot','orderbook_delta','trade');ch=Chains()
    try:
        with op(path,'rt') as f:
            for line in f:
                try:m=json.loads(line)
                except ValueError:
                    if diagnostics is not None:diagnostics['bad_json_lines']+=1
                    continue
                if m.get('type') in kinds and (m.get('msg') or {}).get('market_ticker') in tickers:
                    st.add(m,chain=ch.chain(m.get('sid'),m.get('seq')) if m['type']!='trade' else 0)
                    if m['type']=='orderbook_delta' and len(st)%500==0 and m.get('received_at'):
                        st.lag_sample.append(datetime.fromisoformat(m['received_at'].replace('Z','+00:00')).timestamp()-st.ts[-1])
    except (EOFError,OSError) as ex:
        if diagnostics is not None:diagnostics['truncated_stream']=repr(ex)
    xs=sorted(st.lag_sample);st.lag_median=xs[len(xs)//2] if xs else math.inf
    return st.finalize()


class Execution:
    def __init__(self, log, latency=.1, book_age=math.inf, through='full', repair='local'):
        """through: 'full' = a print strictly through our limit fills us in full (price priority: the taker cleared our whole
        level to reach that price); 'volume' = it fills at most the print's size, shared with other orders (conservative
        sensitivity). repair: 'local' = repair a bad level in place and pause the ticker 5 s; 'strict' = the ticker stays
        invalid until the next snapshot."""
        self.log,self.latency,self.book_age=log,latency,book_age;self.through,self.repair=through,repair
        self.now=0.;self.levels={};self.valid={};self.last={};self.used={}
        self.orders=[];self.busy={};self.rest=[];self.lots=[];self.jobs=[];self.serial=0
        self.stats=collections.Counter();self.ref=lambda tk,t:None
        self.trade_ids=set();self.stream=EventStream();self.si=0;self.suspect={};self.tomb={}

    def attach(self,stream):
        """An EventStream (exchange-time sorted) replayed as the clock advances."""
        self.stream=stream;self.si=0

    def emit(self,event,**kw):
        self.log('execution.jsonl',dict(event=event,exchange_clock=self.now,**kw))

    def schedule(self,t,kind,obj):
        self.serial+=1;heapq.heappush(self.jobs,(t,self.serial,kind,obj))

    def advance(self,t):
        assert t+EPS>=self.now,('clock_reversal',t,self.now)
        while True:
            nj=self.jobs[0][0] if self.jobs else math.inf
            ne=self.stream.ts[self.si] if self.si<len(self.stream) else math.inf
            if min(nj,ne)>t+EPS:break
            if ne<=nj:
                self.now=max(self.now,ne);self.play(self.si);self.si+=1
            else:
                at,_,kind,obj=heapq.heappop(self.jobs);self.now=max(self.now,at)
                getattr(self,'job_'+kind)(obj)
        self.now=max(self.now,t)

    def play(self,i):
        st=self.stream;k=st.kind[i]
        if k==EventStream.DELTA:
            tk=st.names[st.tk[i]];self.delta(tk,'yes' if st.side[i]==0 else 'no',st.pk[i],st.val[i]);self.check(tk)
        else:self.message(st.payload[st.pk[i]])

    def update(self,mk,now):
        # Engine compatibility: all deadlines live on this event scheduler.
        self.advance(now)

    def is_busy(self,tk):return self.busy.get(tk) is not None

    def position(self,tk):
        p=self.busy.get(tk)
        if p is None:return 0.,0.
        held=sum(x['remaining'] for x in p['lots'])
        return p['side']*(held+p['pending']),p['limit'] if p['pending'] else 0.

    def place(self,d):
        assert not self.is_busy(d['ticker'])
        limit=snap(d['limit'],d['side'])
        p=dict(id=len(self.orders),ticker=d['ticker'],kind=d['kind'],side=d['side'],limit=limit,
               size=d['size'],signal=d['signal'],decision_t=d['t'],t0=self.now,pending=d['size'],lots=[],entry_done=False)
        self.orders.append(p);self.busy[p['ticker']]=p
        self.emit('submitted',id=p['id'],ticker=p['ticker'],size=p['size'],limit=limit,wanted_limit=d['limit'],side=p['side'],decision_t=d['t'])
        self.schedule(self.now+self.latency,'entry',p)

    def healthy(self,tk):
        return self.valid.get(tk,False) and self.now>=self.suspect.get(tk,-math.inf) and self.now-self.last.get(tk,-math.inf)<=self.book_age

    def message(self,m):
        d=m.get('msg') or {};tk=d.get('market_ticker');kind=m.get('type')
        if not tk:return
        if kind=='orderbook_snapshot':
            # Full replacement is essential on reconnect.
            b={'yes':{},'no':{}}
            for sd in b:
                vals=d.get(sd+'_dollars_fp');scale=10000 if vals is not None else 100
                for px,q in vals if vals is not None else (d.get(sd) or []):
                    if float(q)>0:b[sd][round(float(px)*scale)]=float(q)
            self.levels[tk]=b;self.valid[tk]=True;self.last[tk]=self.now;self.suspect.pop(tk,None)
            for k in [k for k in self.tomb if k[0]==tk]:del self.tomb[k]
            for k in list(self.used):
                if k[0]==tk:del self.used[k]
            # An unknown reconnect interval invalidates precise queue knowledge.
            for r in self.rest:
                if r['active'] and r['p']['ticker']==tk:
                    sd,pk=self.level_key(r['side'],r['limit'])
                    r['ahead']=max(r['ahead'],b[sd].get(pk,0.))
            self.stats['snapshots']+=1
        elif kind=='orderbook_delta':
            sd=d['side'];pk=round(float(d['price_dollars'])*10000) if d.get('price_dollars') is not None else round(float(d['price'])*100)
            self.delta(tk,sd,pk,float(d.get('delta_fp',d.get('delta',0))))
        elif kind=='trade':
            tid=d.get('trade_id')
            if tid and tid in self.trade_ids:self.stats['duplicate_trades']+=1;return
            if tid:self.trade_ids.add(tid)
            if d.get('is_block_trade'):self.stats['block_trades_ignored']+=1;return
            if not self.healthy(tk):self.stats['trades_without_healthy_book']+=1;return
            self.trade(tk,d)
        self.check(tk)

    def check(self,tk):
        if tk in self.levels:
            b=self.levels[tk]
            if b['yes'] and b['no'] and max(b['yes'])+max(b['no'])>10000:
                self.suspect[tk]=self.now+SUSPECT_S;self.stats['crossed_book']+=1

    def delta(self,tk,sd,pk,delta):
        if tk not in self.levels:
            self.valid[tk]=False;self.stats['delta_without_snapshot']+=1;return
        book=self.levels[tk];b=book[sd];old=b.get(pk,0.);new=old+delta;key=(tk,sd,pk)
        if new < -EPS:
            # A change we have no level for: normally one we already removed as stale (below); otherwise a missed message.
            if self.tomb.pop(key,None) is None:
                self.suspect[tk]=self.now+SUSPECT_S;self.stats['negative_level']+=1
                if self.repair=='strict':self.valid[tk]=False
        if new>EPS:b[pk]=new
        else:b.pop(pk,None)
        if delta>0:
            # A resting add cannot cross the book, so opposite levels it would cross are stale (their removal was missed
            # or reordered by the timestamp jitter between the two sides).
            other='no' if sd=='yes' else 'yes';ob=book[other]
            for q in [q for q in ob if q+pk>=10000]:
                self.tomb[(tk,other,q)]=ob.pop(q);self.used.pop((tk,other,q),None);self.stats['stale_levels_removed']+=1
                if self.repair=='strict':self.valid[tk]=False
        # What we took cannot exceed what the real level now holds.
        if key in self.used:
            self.used[key]=min(self.used[key],max(new,0.))
            if self.used[key]<=EPS:del self.used[key]
        self.last[tk]=self.now;self.stats['deltas']+=1

    @staticmethod
    def level_key(side,px):return ('yes',round(px*100)) if side==1 else ('no',round((100-px)*100))

    def sweep(self,r,limit=None):
        tk=r['p']['ticker'];side=r['side']
        if not self.healthy(tk):self.stats['unavailable_book_attempts']+=1;return
        sd='no' if side==1 else 'yes';b=self.levels[tk][sd]
        for pk in sorted(b,reverse=True):
            px=(10000-pk)/100 if side==1 else pk/100
            if limit is not None and ((side==1 and px>limit+EPS) or (side==-1 and px<limit-EPS)):break
            key=(tk,sd,pk);available=max(0.,b[pk]-self.used.get(key,0.))
            q=min(r['remaining'],available)
            if q>EPS:
                self.used[key]=self.used.get(key,0.)+q
                self.fill(r,q,px,False)
            if r['remaining']<=EPS:break

    def submit(self,p,side,limit,size,lot=None):
        r=dict(p=p,side=side,limit=limit,remaining=size,lot=lot,active=True,at=self.now,serial=self.serial,ahead=0.,ledger=(FeeLedger(),FeeLedger(CENT)))
        if self.healthy(p['ticker']):
            self.sweep(r,limit)
            sd,pk=self.level_key(side,limit)
            r['ahead']=max(0.,self.levels[p['ticker']][sd].get(pk,0.)-self.used.get((p['ticker'],sd,pk),0.))
            if r['remaining']>EPS:self.rest.append(r)
        else:
            # Unknown state cannot support a plausible accepted resting order.
            r['active']=False;self.stats['rejected_missing_book']+=1
        return r

    def job_entry(self,p):
        if p['entry_done']:return
        p['accepted_t']=self.now
        p['entry_order']=self.submit(p,p['side'],p['limit'],p['pending'])
        self.emit('entry_accepted' if p['entry_order']['active'] else 'entry_rejected',id=p['id'])
        if not p['entry_order']['active']:
            p['pending']=0.;p['entry_done']=True;self.finish(p);return
        self.schedule(self.now+60+self.latency,'cancel_entry',p)

    def job_cancel_entry(self,p):
        p['entry_order']['active']=False;p['pending']=0.;p['entry_done']=True
        self.emit('entry_cancel_ack',id=p['id']);self.finish(p)

    def fill(self,r,q,px,maker):
        q=min(q,r['remaining']);r['remaining']-=q;p=r['p']
        # Buying yes pays px per contract, selling yes receives it; fee in cents for the whole slice.
        revenue=-r['side']*D(repr(round(px,6)))/100*D(repr(round(q,2)))
        mf=model_fee(p['ticker'],maker,px,q)
        fee_c=float(r['ledger'][0].charge(revenue,mf)*100);fee1c_c=float(r['ledger'][1].charge(revenue,mf)*100)
        if r['lot'] is None:
            p['pending']=max(0.,p['pending']-q)
            lot=dict(id=len(self.lots),parent=p['id'],ticker=p['ticker'],kind=p['kind'],side=p['side'],size=q,remaining=q,
                     entry=px,entry_fee_c=fee_c,entry_fee1c_c=fee1c_c,entry_maker=maker,t_fill=self.now,exits=[])
            self.lots.append(lot);p['lots'].append(lot)
            self.schedule(self.now+120+self.latency,'exit',lot)
        else:
            lot=r['lot'];lot['remaining']=max(0.,lot['remaining']-q)
            lot['exits'].append(dict(size=q,price=px,fee_c=fee_c,fee1c_c=fee1c_c,t=self.now,maker=maker))
        self.emit('fill',id=p['id'],leg='entry' if r['lot'] is None else 'exit',size=q,price=px,maker=maker,fee_c=fee_c,fee1c_c=fee1c_c)
        if r['remaining']<=EPS:r['active']=False
        if p['pending']<=EPS:
            p['entry_done']=True
        self.finish(p)

    def trade(self,tk,d):
        px=float(d['yes_price_dollars'])*100 if d.get('yes_price_dollars') is not None else float(d['yes_price'])
        q=float(d.get('count_fp',d.get('count',0)))
        side=-1 if d['taker_side']=='yes' else 1
        # One budget for prints at a price, including across multiple exit lots at different prices.
        rs=sorted((r for r in self.rest if r['active'] and r['p']['ticker']==tk and r['side']==side),
                  key=lambda r:(-side*r['limit'],r['at'],r['serial']))
        for r in rs:
            through=px<r['limit']-EPS if side==1 else px>r['limit']+EPS
            at=abs(px-r['limit'])<EPS
            if through:
                r['ahead']=0.
                if self.through=='full':
                    # Price priority: to print at a worse price the taker cleared every order at our limit, with us at its back.
                    if q<r['remaining']-EPS:self.stats['through_print_smaller_than_order']+=1
                    self.schedule(self.now+.25,'audit',(r,px))
                    self.fill(r,r['remaining'],r['limit'],True);self.stats['through_fills']+=1
                else:
                    take=min(q,r['remaining'])
                    if take>EPS:self.fill(r,take,r['limit'],True);q-=take;self.stats['through_fills']+=1
                continue
            if not at or q<=EPS:continue
            ahead=min(q,r['ahead']);r['ahead']-=ahead;q-=ahead
            for later in rs:
                if later is not r and later['limit']==r['limit'] and later['at']>=r['at']:
                    later['ahead']=max(0.,later['ahead']-ahead)
            take=min(q,r['remaining'])
            if take>EPS:self.fill(r,take,r['limit'],True);q-=take
        self.stats['trades']+=1
        if len(self.rest)>1000:self.rest=[r for r in self.rest if r['active']]

    def job_audit(self,info):
        """0.25 s after a through-fill: does the recorded book agree that our level was consumed? Counts levels at our price
        or better that still hold size (new orders joining after the clearing also show up here)."""
        r,px=info;tk=r['p']['ticker'];sd,pk=self.level_key(r['side'],r['limit']);b=self.levels.get(tk,{}).get(sd,{})
        self.stats['through_audit_level_empty' if not any(p>=pk for p in b) else 'through_audit_level_has_size']+=1

    def job_exit(self,lot):
        if lot['remaining']<=EPS:return
        p=self.orders[lot['parent']];ref=self.ref(lot['ticker'],self.now)
        if ref is None:ref=lot['entry']
        limit=snap(ref+p['side'],-p['side'])
        lot['exit_order']=self.submit(p,-p['side'],limit,lot['remaining'],lot)
        self.emit('exit_accepted',id=p['id'],lot=lot['id'],limit=limit)
        self.schedule(self.now+120+self.latency,'taker',lot)

    def job_taker(self,lot):
        if lot['remaining']<=EPS:return
        old=lot.get('exit_order')
        if old:old['active']=False
        self.market_exit(lot)
        if lot['remaining']>EPS:
            lot['taker_end']=self.now+120
            self.schedule(self.now+1,'retry',lot)

    def market_exit(self,lot):
        p=self.orders[lot['parent']]
        r=dict(p=p,side=-p['side'],limit=None,remaining=lot['remaining'],lot=lot,active=True,ledger=(FeeLedger(),FeeLedger(CENT)))
        self.sweep(r)

    def job_retry(self,lot):
        if lot['remaining']<=EPS:return
        if self.now>lot['taker_end']+EPS:
            self.emit('to_settlement',lot=lot['id'],size=lot['remaining']);return
        self.market_exit(lot)
        if lot['remaining']>EPS:self.schedule(self.now+1,'retry',lot)

    def finish(self,p):
        if p['entry_done'] and all(x['remaining']<=EPS for x in p['lots']):
            if self.busy.get(p['ticker']) is p:self.busy[p['ticker']]=None

    def cancel_entries(self,t):
        for p in self.orders:
            if not p['entry_done']:self.schedule(max(self.now,t)+self.latency,'cancel_entry',p)

    def results(self,payoff):
        by={k:dict(entry_contracts=0.,fills=0,pnl_c=0.,fees_c=0.,pnl_1c_c=0.,fees_1c_c=0.,settled_contracts=0.) for k in ('spread','total')}
        for lot in self.lots:
            a=by[lot['kind']];a['entry_contracts']+=lot['size'];a['fills']+=1
            gross=0.
            for x in lot['exits']:gross+=x['size']*lot['side']*(x['price']-lot['entry'])
            if lot['remaining']>EPS:
                a['settled_contracts']+=lot['remaining']
                gross+=lot['remaining']*lot['side']*(payoff(lot['ticker'])-lot['entry'])
            for tag,entry,fees_k,pnl_k in (('fee_c','entry_fee_c','fees_c','pnl_c'),('fee1c_c','entry_fee1c_c','fees_1c_c','pnl_1c_c')):
                fees=lot[entry]+sum(x[tag] for x in lot['exits'])
                a[fees_k]+=fees;a[pnl_k]+=gross-fees
        for a in by.values():
            a['net_per_contract_c']=a['pnl_c']/a['entry_contracts'] if a['entry_contracts'] else None
            a['gross_c']=a['pnl_c']+a['fees_c']
        return dict(orders=len(self.orders),by_kind=by,diagnostics=dict(self.stats))

    def export_lots(self):
        # exit_order points back to its parent/lot and is execution state,
        # not a serializable fill record.
        return [{k:v for k,v in lot.items() if k!='exit_order'} for lot in self.lots]
