# Simple Mean-Variance Portfolio Backtester

## Project Overview

This project implements a rolling-window mean-variance portfolio backtesting system, comparing strategies such as Equal Weight, Buy & Hold, and Markowitz optimization using real U.S. stock market data.

## Key Features

Fetch historical data for multiple stocks via yfinance

Implement rolling-window Mean-Variance Optimization

Support periodic rebalancing

Comprehensive performance evaluation including annualized return, volatility, Sharpe ratio, and maximum drawdown

Visualization of net asset value curves and drawdowns

<<<<<<< HEAD
=======
## Strategy Details

#### 1.Buy & Hold

Initial weights are set equally.
No rebalancing is performed throughout the backtest period.

#### 2.Equal Weight

Starts with equal weights.
Rebalances back to equal weights every 20 trading days.

#### 3.Mean-Variance Optimization

Lookback window: 60 trading days
Rebalancing frequency: every 20 trading days
Optimization objective: Maximize Sharpe ratio
Constraints: weights sum to 1 and each weight is from 0 to 1 (long-only, no short-selling)
Mean returns and covariance matrix are annualized using 252 trading days.
Risk-free rate is fixed at 2% per year.

>>>>>>> 8d01ce6 (update the readme file and correct some mistakes)
## Main Results

### Drawdown Comparison

![alt text](results/drawdown_comparison.png)
<<<<<<< HEAD
The Buy & Hold and Equal Weight strategies exhibit highly comparable drawdown dynamics, reflecting similar risk profiles during the observed period. In contrast, the Markowitz portfolio demonstrates a more favorable drawdown trajectory, achieving a minimum drawdown rate of −0.2101, thereby indicating superior downside risk management relative to the benchmark strategies.
=======
The Mean-Variance strategy shows better downside control, with a maximum drawdown of approximately -21.0%, outperforming both Buy & Hold and Equal Weight strategies.
>>>>>>> 8d01ce6 (update the readme file and correct some mistakes)

### Net Value Comparison

![alt text](results/net_value_comparison.png)
<<<<<<< HEAD
The Markowitz strategy demonstrates a superior trajectory in net asset value growth relative to the benchmark strategies. Between January 2022 and May 2023, the Buy & Hold and Equal Weight portfolios exhibit highly similar performance patterns. However, subsequent to May 2023, the Buy & Hold strategy begins to outperform the Equal Weight approach, with the performance gap between the two strategies widening progressively over time.

(During the initial 60-day period, the Markowitz strategy exhibits performance similar to the other two approaches because its backlook window is set to 60 days and the initial weight parameters are consistent with those of the other two groups. )

## Limitation

The backtesting framework employed in this study does not incorporate transaction fees, which constitutes a methodological limitation. Consequently, the reported net asset values and returns are likely to be upwardly biased relative to real-world outcomes, as the omission of trading costs may overstate the attainable performance.
For computational simplicity, the Sharpe ratio is calculated using a fixed risk-free rate of 0.02, which may introduce deviations from the actual values.
=======
The Mean-Variance portfolio achieves a higher terminal net value compared with the two benchmark strategies over the test period.

## Limitation

Transaction costs and market impact are not included. Therefore, reported returns and net asset values are likely overstated relative to live trading.
The risk-free rate is assumed constant at 2%.
Covariance matrix is estimated using a simple rolling window without shrinkage or other robust techniques, which may lead to estimation error.
The stock universe is small and consists of large-cap U.S. stocks, so results may not generalize to broader markets.
>>>>>>> 8d01ce6 (update the readme file and correct some mistakes)

## Tech Stack

Python, Pandas, NumPy, SciPy, Matplotlib, yfinance
