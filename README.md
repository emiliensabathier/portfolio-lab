# portfolio-lab

Multi-asset portfolio construction, risk analytics and backtesting on a ten-ETF universe.

![ci](https://github.com/emiliensabathier/portfolio-lab/actions/workflows/ci.yml/badge.svg)

![Growth of one unit of capital, net of costs, for each strategy against the 60/40 benchmark](docs/growth-of-capital.png)

**In short**

- None of equal weight, minimum variance, risk parity or maximum Sharpe beats a static 60/40 (Sharpe 0.82) out of sample from 2010 to 2026, net of 5 bps costs.
- Only equal weight differs at the 5% level (Jobson-Korkie-Memmel p = 0.038, and it is worse); nothing survives Bonferroni.
- Look-ahead is impossible by construction: a strategy only ever receives a copy of the past, and a spy test checks it.

Rendered report: <https://emiliensabathier.github.io/portfolio-lab/>

## Results

Out-of-sample 2010-08-02 to 2026-08-14, monthly rebalanced, **net of 5 bps transaction
costs**. Sharpe and Sortino are **excess of the three-month Treasury bill** (FRED `DTB3`,
converted to the bond-equivalent yield), which averaged 1.55% over this window and ranged
from -0.05% to 5.51%. The last column is the p-value of a test that the rule's Sharpe
equals the benchmark's (see below).

| Strategy | CAGR | Volatility | Sharpe | Sortino | Max drawdown | Turnover p.a. | p vs 60/40 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 60/40 benchmark | 9.93% | 10.32% | 0.82 | 1.15 | -21.63% | 0.18 | — |
| Equal weight | 7.40% | 10.47% | 0.59 | 0.81 | -23.21% | 0.29 | 0.038 |
| Minimum variance | 3.53% | 4.61% | 0.44 | 0.60 | -15.11% | 0.35 | 0.130 |
| Risk parity | 5.68% | 8.17% | 0.53 | 0.73 | -20.26% | 0.45 | 0.080 |
| Maximum Sharpe | 7.95% | 10.07% | 0.66 | 0.90 | -21.21% | 2.14 | 0.511 |

Not one of the four rules beats a static 60/40 on risk-adjusted return, and measuring
against cash rather than against zero is what makes that legible: it costs every strategy
between 0.15 and 0.34 of Sharpe, and it costs the low-volatility rules most, because a
minimum-variance book earning 3.53% while bills paid 1.55% has given up most of its edge
to the riskless leg.

What the data supports is "does not beat", not "is worse than". A two-sided
Jobson-Korkie test with the Memmel correction, on daily excess returns, rejects equal
Sharpe ratios at 5% only for equal weight (p = 0.038); for the other three rules the
shortfall against 60/40 is within what sixteen years of daily noise can produce. Four
tests are run, so after a Bonferroni correction (1.25% per test) none of the shortfalls
is significant.

The benchmark is the top line and it is not close. That is the result: four textbook
allocation rules, each of them defensible, none of them beating the simplest possible
portfolio over this window. The chart at the top of this page is drawn from the frozen fixture, which ends
2024-12-30, so it stops earlier than the table above.

Full report with charts, tail risk and stress tests: [`reports/portfolio.html`](reports/portfolio.html).

These figures come from `python -m plab --end 2026-08-14` and will drift as more history
accumulates. `tests/fixtures/expected_metrics.json` freezes the same computation to
2024-12-30, so a reader can reproduce that snapshot offline, without network access, even
as the numbers above keep moving.

## Method

- **Universe**: ten liquid ETFs across equity (SPY, IWM, EFA, EEM), rates (AGG, TLT, HYG)
  and real assets (GLD, VNQ, DBC). The list was chosen in 2026; see Limitations for what
  that implies.
- **Estimation**: 36-month rolling window, Ledoit-Wolf shrinkage towards a scaled identity
  target, implemented from the 2004 paper rather than imported.
- **Rebalancing**: monthly, at the last trading day. Weights decided on day *d* apply from
  day *d+1*.
- **Costs**: 5 bps charged on turnover, where turnover is the sum of absolute weight
  changes. Every figure in the results table above is net.
- **Benchmarks**: 60/40 SPY/AGG and equal weight. No strategy is reported without both.
- **Significance**: each rule's Sharpe is tested against the 60/40's with Jobson-Korkie
  (1981) as corrected by Memmel (2003), which accounts for the correlation between the
  two return series.

### Look-ahead bias

A strategy is a function `(date, history) -> weights`. The engine hands it a copy of the
trailing estimation window ending at `date` and nothing else, so a strategy cannot read the
future even if its author tries to. `tests/test_engine.py` contains a spy strategy that
records the latest observation it was shown; the test asserts it never exceeds the decision
date. The guarantee is enforced by the interface, not by convention.

## Limitations

Stated because they matter more than the headline numbers.

- **The universe was picked with hindsight.** All ten ETFs still trade today, and they were
  chosen in 2026 by someone who knew which asset classes and which funds had lasted. That
  is a selection bias, and a look-ahead in the choice of universe, even though the engine
  prevents look-ahead in the weights. Broad index ETFs rather than single stocks limit
  survivorship inside each fund, since an index replaces its own failed constituents, but
  they do nothing about the bias in choosing the funds.
- **The 2008 crisis is not in the out-of-sample backtest.** HYG listed in April 2007, so the
  common history starts 2007-07; the first 36 months are consumed by the estimation window,
  so the backtest itself only begins 2010-07 (first weights decided 2010-07-30, first return
  recorded 2010-08-02).
- **The stress tests are how 2008 is covered**, by applying each strategy's *final* weights
  to the realized asset returns of a historical crisis window. These are counterfactuals on
  the current allocation, not performance the strategy lived through: the weights are held
  constant across the whole window, equivalent to rebalancing back to them daily at no cost.
  Unlike every other figure in this repository, the stress figures are **not** net of
  transaction costs.
- **The stress figures depend on when you run them**, because they replay the strategy's
  *final* weights, and those weights change as new history arrives. Concretely: the live run
  behind the results table above puts the 2008 loss for `min_variance` at **-13.85%**, while
  the frozen fixture ending 2024-12-30 gives **-4.5%** for the same strategy and the same
  scenario. Same code, same method, different as-of date — a reader who compares the two
  without this note would reasonably conclude something is broken.
- **No slippage or market impact model.** Transaction costs are a flat spread on turnover.
  Realistic for ten large ETFs at small size, optimistic at scale.
- **Maximum Sharpe estimates expected returns as trailing sample means.** Each month it
  maximizes the excess-of-bill Sharpe estimated on the previous 36 months and holds the
  result for the next month, so the weights are out-of-sample but rest on the noisiest
  input in portfolio construction; its turnover of 2.14 a year is that noise. It is
  included to make the instability visible next to the other rules, not as a
  recommendation.
- **The Sharpe test assumes independent, normal returns.** The Jobson-Korkie-Memmel
  variance is derived under that assumption; daily returns are fat-tailed and their
  volatility clusters, which can make the test too ready to reject. The p-values are
  indicative, and the conclusion they support (no rule beats 60/40) does not hinge on
  their precision.
- **The riskless leg is the three-month bill, and it is a bill, not the funding rate any
  particular investor faces.** FRED's `DTB3` is quoted on a bank-discount basis; `cash.py`
  converts it once to the bond-equivalent yield, which is the number an investor earns on
  the price paid, so the conversion is removed rather than disclosed. What remains
  disclosed: a real book funds at a spread to bills, not at bills, and no such spread is
  modelled here. The series used to be Yahoo's `^IRX`, the CBOE index on the 13-week bill,
  read through `yfinance`, an unofficial scraper. It moved to FRED for provenance: `DTB3`
  is the Federal Reserve's own H.15 series at a stable, documented endpoint. Over this
  window the two differ by about 1.5 basis points on average.
- **Dividends** are handled through adjusted close prices, which assumes reinvestment at
  close with no tax.
- **The Cornish-Fisher expansion degrades on daily returns with high excess kurtosis.**
  The correction is a truncated series around the Gaussian quantile; on a fat-tailed
  sample the quartic kurtosis term can dominate and the reported figure can exceed the
  empirical (historical) VaR at the same level by 2-3x. Past a kurtosis threshold the
  expansion leaves its region of validity altogether and the adjusted quantile can flip
  sign, reporting a gain where the sample clearly has a loss. `cornish_fisher_var` checks
  for this and raises rather than return a nonsensical figure — it is not, in general, a
  safe substitute for the historical VaR/ES pair on real portfolio return series.

## Running it

```bash
python -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/python -m plab --output reports/portfolio.html
```

Prices are cached under `cache/`; pass `--refresh` to re-download.

The price panel comes from Yahoo through `yfinance`, and a session can come back missing
for one ETF and present for the rest. Nothing is forward-filled to paper over it, so the
run refuses and names the dates. Pass `--end` to run on the intact history instead; the
date is inclusive, and it cuts the bill series to the same window:

```bash
.venv/bin/python -m plab --end 2026-08-14 --output reports/portfolio.html
```

## Tests

```bash
.venv/bin/python -m pytest --cov=src/plab
```

The suite runs offline against a frozen price fixture, so the published numbers are
reproducible without network access. The bill series is frozen alongside the prices and the
regression test runs the pipeline with it, because pinning a Sharpe measured against zero
while the report shows one measured against cash would go green on numbers nobody
publishes.

## Related

Four companion studies, same method: a frozen capture, a rendered report, and a
limitations section longer than the results.

- [rates-lab](https://github.com/emiliensabathier/rates-lab) — what the yield curve prices: policy path, inflation, term premium
- [valuation-lab](https://github.com/emiliensabathier/valuation-lab) — what a share price already assumes, by inverting a DCF
- [credit-lab](https://github.com/emiliensabathier/credit-lab) — which default score flags first, against real credit events
- [options-lab](https://github.com/emiliensabathier/options-lab) — what S&P 500 implied volatility prices: an arbitrage-free surface and the variance premium

## License

MIT.
