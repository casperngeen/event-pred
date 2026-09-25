/Users/caspe2/NUS/FYP/event-pred/../analysis/exploratory_2026_08/spread_measurement.py:17: SyntaxWarning: invalid escape sequence '\{'
  LC_ALL=C grep -E '^\{"market": "(KXFEDDECISION|RECSSNBER)' \

==============================================================================
BOOK COVERAGE — the whole macro sample in kalshi_orderbooks.jsonl
==============================================================================
shape: (3, 4)
┌─────────────────────────┬───────────┬─────────────────────────┬─────────────────────────┐
│ ticker                  ┆ snapshots ┆ from                    ┆ to                      │
│ ---                     ┆ ---       ┆ ---                     ┆ ---                     │
│ str                     ┆ u32       ┆ datetime[μs]            ┆ datetime[μs]            │
╞═════════════════════════╪═══════════╪═════════════════════════╪═════════════════════════╡
│ KXFEDDECISION-26JAN-H0  ┆ 117769    ┆ 2025-10-29 07:30:09.618 ┆ 2025-12-02 19:50:12.559 │
│ RECSSNBER-25            ┆ 7203      ┆ 2025-10-29 07:27:24.952 ┆ 2025-10-30 21:41:05.731 │
│ KXFEDDECISION-26JAN-C25 ┆ 2         ┆ 2025-10-29 07:30:09.617 ┆ 2025-10-29 07:37:55.362 │
└─────────────────────────┴───────────┴─────────────────────────┴─────────────────────────┘

Everything below rests on KXFEDDECISION-26JAN-H0: one contract, 117,769 snapshots,
and it is FAR-DATED throughout (56-90 days to close).

==============================================================================
1  UNCONDITIONAL vs EXECUTABLE SPREAD
==============================================================================
all snapshots        : mean 8.30c  median 8.0c
time-weighted        : 7.18c
percentiles          : p5=1  p10=2  p25=4  p50=8  p75=11  p90=13  p95=14

AT TRADE MOMENTS (n=121 of 125 trades matched):
  mean 4.69c  median 4.0c  volume-weighted 5.11c
  percentiles        : p10=1  p25=2  p50=4  p75=7  p90=9
  P(spread<=2c|trade) 0.355  vs unconditional 0.127
  P(spread<=4c|trade) 0.579  vs unconditional 0.253
  effective half-spread paid, taker buys yes: mean 2.74c  median 2.0c  (n=70)
  effective half-spread paid, taker buys no : mean 3.21c  median 3.5c  (n=51)

Selection bias is real but ~1.8x, not 8x: trades cluster at tight moments.

==============================================================================
2  DOES THE SPREAD TIGHTEN TOWARD RESOLUTION?
==============================================================================
(a) within-contract, KXFEDDECISION-26JAN-H0 (only 56-91 DTC observed):
    days to close         n    mean  median    mid
    90-95             2215   10.28      11     50
    85-90            34433   12.82      11     59
    80-85            13614    7.40       8     60
    75-80            14399    7.29       7     60
    70-75            14400   10.31       7     57
    65-70            13291    6.68       8     48
    60-65            14397    2.87       3     60
    55-60            11020    2.64       3     66

(b) cross-section of live quotes, one per ticker, fetched pre-2026:

  MACRO: n=507, median spread 6c
    bucket            n  med spr  mean spr   med OI
    1-3d              8        8      7.88       28
    3-7d              8        9      8.62        0
    7-14d            17        2      2.53     1578
    14-30d           28        2      3.75     8220
    30-90d           73        6      7.96       13
    90d+            373        6      6.37        0

  ALL MARKETS: n=7200, median spread 6c
    bucket            n  med spr  mean spr   med OI
    <1d             261       28     41.48        0
    1-3d             68        6     13.99        6
    3-7d             75        7      7.17       40
    7-14d           112        3      4.53     6946
    14-30d          531        4     20.96       48
    30-90d         1998        5     13.98     2644
    90d+           4155        6      8.73      519

  Open interest tracks the spread more closely than days-to-close does:
  the tight buckets are the ones with real OI, not simply the near ones.
  Caveat: one moment in time, one quote per ticker, small macro n near close.

==============================================================================
3  BREAK-EVENS RECOMPUTED — perfect foresight, single leg
==============================================================================
E|R| from surprise_cost_test.py test C: dormant 2.39c, liquid 8.43c

spread basis                              spread   +fees   dormant   liquid
unconditional book                          8.30   11.80     -9.41    -3.37
time-weighted book                          7.18   10.68     -8.29    -2.25
at trade moments (mean)                     4.69    8.19     -5.80     0.24
at trade moments (median)                   4.00    7.50     -5.11     0.93
2x effective half-spread paid               5.48    8.98     -6.59    -0.55
trade-bounce estimate (S4.2.1)              1.00    4.50     -2.11     3.93

==============================================================================
4  LADDER WIDTH — how many legs an implied-mean trade actually needs
==============================================================================
median legs/event 10  mean 11.0  p90 15

  WTI            events  702  median legs 15
  WTIW           events  161  median legs 15
  JOBLESS        events   69  median legs 1
  CPI            events   54  median legs 8
  U3             events   53  median legs 9
  CPICORE        events   42  median legs 7
  CPIYOY         events   37  median legs 11
  CPICOREYOY     events   36  median legs 10
  FED            events   35  median legs 11
  PROLLS         events   24  median legs 6

cost of expressing one implied-mean view across k legs (at 4.69c + 2 fees):
  1 leg(s):   8.19c   dormant PF    -5.80   liquid PF    +0.24
  3 leg(s):  24.57c   dormant PF   -22.18   liquid PF   -16.14
  5 leg(s):  40.95c   dormant PF   -38.56   liquid PF   -32.52
  8 leg(s):  65.52c   dormant PF   -63.13   liquid PF   -57.09
