# portfolio-lab

Multi-asset portfolio construction, risk analytics and backtesting on a ten-ETF universe.

![ci](https://github.com/emiliensabathier/portfolio-lab/actions/workflows/ci.yml/badge.svg)

## Results

Out-of-sample, monthly rebalanced, **net of 5 bps transaction costs**, 2010-07 to today.

| Strategy | CAGR | Volatility | Sharpe | Max drawdown | Turnover p.a. |
| --- | --- | --- | --- | --- | --- |
| 60/40 benchmark | 9.92% | 10.33% | 0.97 | -21.63% | 0.24 |
| Equal weight | 7.34% | 10.48% | 0.73 | -23.21% | 0.35 |
| Minimum variance | 3.51% | 4.62% | 0.77 | -15.11% | 0.41 |
| Risk parity | 5.63% | 8.17% | 0.71 | -20.26% | 0.51 |
| Maximum Sharpe | 7.55% | 9.09% | 0.85 | -21.15% | 2.07 |

Full report with charts, tail risk and stress tests: [`reports/portfolio.html`](reports/portfolio.html).

These figures come from a live run and will drift as more history accumulates.
`tests/fixtures/expected_metrics.json` freezes the same computation to 2024-12-30, so a
reader can reproduce that snapshot offline, without network access, even as the numbers
above keep moving.

## Method

- **Universe**: ten liquid ETFs across equity (SPY, IWM, EFA, EEM), rates (AGG, TLT, HYG)
  and real assets (GLD, VNQ, DBC). ETFs rather than single stocks, because an ETF does not
  disappear: the universe carries **no survivorship bias**.
- **Estimation**: 36-month rolling window, Ledoit-Wolf shrinkage towards a scaled identity
  target, implemented from the 2004 paper rather than imported.
- **Rebalancing**: monthly, at the last trading day. Weights decided on day *d* apply from
  day *d+1*.
- **Costs**: 5 bps charged on turnover, where turnover is the sum of absolute weight
  changes. Every figure in the results table above is net.
- **Benchmarks**: 60/40 SPY/AGG and equal weight. No strategy is reported without both.

### Look-ahead bias

A strategy is a function `(date, history) -> weights`. The engine hands it a copy of the
trailing estimation window ending at `date` and nothing else, so a strategy cannot read the
future even if its author tries to. `tests/test_engine.py` contains a spy strategy that
records the latest observation it was shown; the test asserts it never exceeds the decision
date. The guarantee is enforced by the interface, not by convention.

## Limitations

Stated because they matter more than the headline numbers.

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
  behind the results table above puts the 2008 loss for `min_variance` at **-13.97%**, while
  the frozen fixture ending 2024-12-30 gives **-4.5%** for the same strategy and the same
  scenario. Same code, same method, different as-of date — a reader who compares the two
  without this note would reasonably conclude something is broken.
- **No slippage or market impact model.** Transaction costs are a flat spread on turnover.
  Realistic for ten large ETFs at small size, optimistic at scale.
- **Maximum Sharpe uses in-sample expected returns** and is included precisely to make its
  instability visible next to the other rules, not as a recommendation.
- **Dividends** are handled through adjusted close prices, which assumes reinvestment at
  close with no tax.

## Running it

```bash
python -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/python -m plab --output reports/portfolio.html
```

Prices are cached under `cache/`; pass `--refresh` to re-download.

## Tests

```bash
.venv/bin/python -m pytest --cov=src/plab
```

The suite runs offline against a frozen price fixture, so the published numbers are
reproducible without network access.

## License

MIT.
