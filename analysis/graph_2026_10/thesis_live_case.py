"""Render the live MIL--LAD paper-order case from recorded decisions and trades."""
from pathlib import Path
import datetime as dt
import gzip
import json
import os
import tempfile

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'event_pred_matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT/'data/sports_live/sandbox/2026-10-15/MLB_849810'
GEN = ROOT/'thesis/generated'
TICKER = 'KXMLBTOTAL-26OCT152000MILLAD-10'


def records(name):
    with gzip.open(SOURCE/name, 'rt') as stream:
        for line in stream:
            yield json.loads(line)


def main():
    utc = dt.timezone.utc
    date = lambda stamp: dt.datetime.fromtimestamp(stamp, utc)
    start = dt.datetime(2026, 10, 16, 2, 56, 15, tzinfo=utc).timestamp()
    stop = start+80
    decisions = [r for r in records('decisions.jsonl.gz')
                 if r.get('ticker') == TICKER and start <= r['t'] <= stop]
    order = next(r for r in decisions if r.get('action') == 'order')
    events = [r for r in records('orders.jsonl.gz') if r.get('id') == 9]
    fill = next(r for r in events if r['event'] == 'filled')
    assert order['limit'] == fill['price'] == 50 and order['side'] == -1
    trades = {}
    for r in records('trades.jsonl.gz'):
        if r.get('ticker') != TICKER:
            continue
        t = dt.datetime.fromisoformat(r['created_time']).timestamp()
        if start <= t <= stop:
            trades[r['trade_id']] = (t, 100*float(r['yes_price_dollars']))
    prices = sorted(trades.values())
    plt.rcParams.update({'font.size': 9, 'font.family': 'serif'})
    fig = plt.figure(figsize=(7.1, 4.7))
    ax = fig.add_axes([.09, .40, .88, .54])
    ax.step([date(t) for t, _ in prices], [p for _, p in prices], where='post',
            color='#33658a', linewidth=1.2, label='Recorded trade price')
    ax.plot([date(r['t']) for r in decisions], [r['level'] for r in decisions],
            'o--', color='#c46c35', markersize=3, linewidth=1,
            label='120 s model forecast')
    ax.plot([date(order['t']), date(fill['t'])], [50, 50], color='#a02c2c',
            linewidth=1.6, label='Resting sell limit')
    ax.scatter([date(fill['t'])], [50], color='#a02c2c', s=35, zorder=5)
    ax.annotate('Sell at 50\nforecast 42.98', (date(order['t']), 50),
                xytext=(date(order['t']+6), 68), arrowprops={'arrowstyle':'->'}, fontsize=8)
    ax.annotate('Home-run repricing\npaper fill at 50', (date(fill['t']), 50),
                xytext=(date(fill['t']-22), 87), arrowprops={'arrowstyle':'->'}, fontsize=8)
    ax.set(xlim=(date(start), date(stop)), ylim=(15, 105), ylabel='YES price (cents)',
           xlabel='16 October 2026, UTC')
    ax.xaxis.set_major_locator(mdates.SecondLocator(bysecond=[0, 15, 30, 45]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S', tz=utc))
    ax.grid(axis='y', alpha=.2)
    ax.legend(loc='upper left', fontsize=7, frameon=False)
    flow = fig.add_axes([.03, .02, .95, .23])
    flow.set(xlim=(0, 1), ylim=(0, 1))
    flow.axis('off')
    labels = ['02:56:27\nThin offers swept\nLast trade: 49',
              '02:56:27.7\nForecast: 43\nSell limit: 50',
              'Next 44 seconds\nMarket falls to 28\nOrder stays unfilled',
              '02:57:12 onward\nHome run; fill at 50\nLater settles at 100']
    for i, label in enumerate(labels):
        x = .125+.25*i
        flow.text(x, .5, label, ha='center', va='center', fontsize=8,
                  bbox=dict(boxstyle='round,pad=.5', facecolor='#f3f5f7', edgecolor='#9aa3aa'))
        if i < 3:
            flow.annotate('', xy=(x+.155, .5), xytext=(x+.095, .5),
                          arrowprops=dict(arrowstyle='->', color='#555555'))
    fig.savefig(GEN/'live_case_timeline.pdf')
    fig.savefig(Path(tempfile.gettempdir())/'thesis_live_case.png', dpi=150)
    (GEN/'live_case_audit.json').write_text(json.dumps(dict(
        source=str(SOURCE.relative_to(ROOT)), ticker=TICKER, decision=order,
        order_events=events, market_trades=len(prices)), indent=2)+'\n')


if __name__ == '__main__':
    main()
