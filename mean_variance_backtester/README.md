# Simple Mean-Variance Portfolio Backtester

## Project Overview

This project implements a rolling-window mean-variance portfolio backtesting system, comparing strategies such as Equal Weight, Buy & Hold, and Markowitz optimization using real U.S. stock market data.

## Key Features

Fetch historical data for multiple stocks via yfinance

Implement rolling-window Mean-Variance Optimization

Support periodic rebalancing

Comprehensive performance evaluation including annualized return, volatility, Sharpe ratio, and maximum drawdown

Visualization of net asset value curves and drawdowns

## Main Results

### Drawdown Comparison

![alt text](results/drawdown_comparison.png)
The Buy & Hold and Equal Weight strategies exhibit highly comparable drawdown dynamics, reflecting similar risk profiles during the observed period. In contrast, the Markowitz portfolio demonstrates a more favorable drawdown trajectory, achieving a minimum drawdown rate of −0.2101, thereby indicating superior downside risk management relative to the benchmark strategies.

### Net Value Comparison

![alt text](results/net_value_comparison.png)
The Markowitz strategy demonstrates a superior trajectory in net asset value growth relative to the benchmark strategies. Between January 2022 and May 2023, the Buy & Hold and Equal Weight portfolios exhibit highly similar performance patterns. However, subsequent to May 2023, the Buy & Hold strategy begins to outperform the Equal Weight approach, with the performance gap between the two strategies widening progressively over time.

## Limitation

The backtesting framework employed in this study does not incorporate transaction fees, which constitutes a methodological limitation. Consequently, the reported net asset values and returns are likely to be upwardly biased relative to real-world outcomes, as the omission of trading costs may overstate the attainable performance.

## Tech Stack

Python, Pandas, NumPy, SciPy, Matplotlib, yfinance
