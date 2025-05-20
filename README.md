# Portfolio Optimization with Reinforcement Learning and Convex Risk Measures

This project implements a reinforcement learning approach to portfolio optimization based on Nassim Taleb's principles of risk management and robust portfolio construction.

## Overview

We use Proximal Policy Optimization (PPO) to train an agent that dynamically adjusts portfolio weights across a diverse set of assets. The agent optimizes a custom reward function that balances:

1. Expected returns (E[r])
2. Conditional Value at Risk (CVaR) - a coherent risk measure preferred by Taleb
3. Portfolio diversity

## Taleb-Inspired Asset Selection

Following Nassim Taleb's principles, we selected assets that create a "barbell strategy" portfolio:

- Safe, highly liquid instruments (BIL, SHV)
- Anti-fragile assets that may benefit from volatility (GLD, ETH-USD)
- Defensive stocks with stable cash flows (JNJ, PG, NOC)
- Growth-oriented assets with potential for higher returns (TSLA, AAPL, ARKK)

This combination aims to be robust against black swan events while still capturing upside potential.

## Reinforcement Learning Environment

The portfolio environment is modeled as a Gym environment with:

- **State**: Historical returns of all assets plus current CVaR and static features
- **Action**: Portfolio weight allocations (0-1 for each asset)
- **Reward**: A linear combination of expected return, CVaR, and diversity

```
reward = E[r] - λ * CVaR + diversity_coeff * diversity
```

Where:
- λ = 0.8 (higher weight on risk management)
- diversity_coeff = 0.2 (encouraging diversification)

## Training Methodology

We employed a walk-forward expanding window approach:

1. Split historical data (2018-2024) into sequential blocks with 80/20 train/test splits
2. For each window:
   - Extract static features from price data
   - Train model on expanding historical data
   - Fine-tune with multiple iterations
   - Evaluate on test period
3. Final test on 2025 data in 15-day windows

Training parameters:
- Pretrain: 1,000 timesteps
- Fine-tune: 800 timesteps per window
- Network architecture: [128, 128] hidden layers
- Max weight adjustment per step: 5%

## Key Findings

1. The combination of E[r] and CVaR in the reward function produces more stable portfolio allocations than traditional mean-variance optimization
2. Limiting position changes (max 5% per step) improves long-term performance
3. The model successfully balances risk and return during volatile market conditions
4. Static features extracted from historical price data improve model performance

## Usage

The project contains the following components:

1. `PortfolioEnv` class: Custom Gym environment for portfolio optimization
2. Training loop with walk-forward validation
3. Evaluation function to test model performance
4. Feature extraction module

To run the model:

```bash
python portfolio_optimization.py
```

## Dependencies

- pandas, numpy
- gym, stable-baselines3
- yfinance
- matplotlib

## Future Work

1. Incorporate additional Taleb-inspired risk measures (e.g., power law tail exponents)
2. Test with different asset classes and market regimes
3. Include transaction costs in the optimization
4. Implement portfolio stress testing
