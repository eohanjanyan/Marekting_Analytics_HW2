# Bandit Algorithms: Epsilon-Greedy and Thompson Sampling

## Overview

This project implements two popular multi-armed bandit algorithms:

- **Epsilon-Greedy**: Balances exploration and exploitation with a decaying epsilon value.
- **Thompson Sampling**: Uses probabilistic models to select the arm with the highest expected reward.

Both algorithms are used to solve the exploration vs. exploitation trade-off by selecting the best action from a set of options to maximize cumulative reward over time.

## Features

- Simulates multiple bandit arms with different reward distributions.
- Implements Epsilon-Greedy and Thompson Sampling algorithms.
- Visualizes the performance of both algorithms.
- Logs trial results to a CSV file.

## Requirements

- `loguru` (for logging)
- `numpy` (for numerical calculations)
- `matplotlib` (for plotting results)
- `scipy` (for statistical functions)

Install dependencies using:

```bash
pip install -r requirements.txt
```

I shall not be responsible for the code not running if you do not install the requirements. 
:)

## Running the Code

1. Clone the repository or download bandit.py

2. Run it using this command.


    ```bash
    python bandit.py
    ```

This will execute the Epsilon-Greedy and Thompson Sampling algorithms, simulate rewards, log the results to `results.csv`, and generate performance visualizations.

