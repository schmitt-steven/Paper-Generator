## General Information

### Topic
Recursive Backwards Q-Learning (RBQL): A model-based reinforcement learning algorithm for deterministic environments that propagates rewards backwards through an explored state-transition model upon reaching terminal states.

### Hypothesis
RBQL converges to optimal policies significantly faster than standard Q-learning in deterministic, episodic environments by leveraging a persistent world model and backward reward propagation, eliminating the need for repeated visits to update Q-values.

## Section Requirements

### Abstract
3-4 sentences summarizing: the problem (Q-learning inefficiency in deterministic tasks), the solution (RBQL with backward propagation), key results (faster convergence, fewer episodes to optimal policy), and implications.

### Introduction
Explain why standard Q-learning is inefficient for deterministic problems (requires many visits to propagate rewards). Introduce model-based RL as solution. State RBQL's core idea: build transition model during exploration, then BFS backwards from terminal states updating all Q-values in one sweep. Clearly state contributions.

### Related Work
Cover: Q-learning fundamentals, model-based vs model-free RL, Dyna-Q architecture, dynamic programming (value iteration), Monte Carlo methods. Distinguish RBQL from each—emphasize that RBQL uses α=1 (full replacement) and single backward sweep vs iterative updates.

### Methods
Describe RBQL algorithm precisely:
1. Persistent model stores (s, a) → (s', r) transitions
2. Epsilon-greedy exploration with decay
3. On terminal state: build backward graph, BFS from terminal, update Q(s,a) = r + γ·max(Q(s'))

### Results
Experiment Setup:
Comparison: RBQL vs. Standard Q-Learning.
Runs: N=30 independent runs per algorithm (Seeds 0–29).
Termination: Max 500 episodes or until convergence.
Convergence Metric: First episode where avg reward $\ge 0.9$ (over rolling window of 10).

Data Collection & Analysis:
Metric 1 (Efficiency): Episodes to convergence.
Metric 2 (Cost): Wall-clock time (seconds) to convergence.
Significance: Compute Welch’s t-test ($p$-value) for both metrics.
Uncertainty: Calculate 95% Confidence Intervals (CI) for all means.

Required Plots:
Learning Curve: Episode vs. Reward. Use shaded 95% CI (not std dev).
Efficiency Frontier (Scatter): Wall-clock time ($x$) vs. Episodes to converge ($y$). Plot all 60 data points.
Significance Bar Chart: Mean episodes to converge with error bars (95% CI) and annotated $p$-values.

### Discussion
Analyze why RBQL outperforms Q-learning in deterministic settings. Discuss limitations: only works for deterministic environments, requires storing full transition model (memory), episodic tasks only. Suggest extensions: stochastic environments (weighted propagation), continuous state spaces, memory-efficient model compression.

### Conclusion
2-3 sentences: RBQL demonstrates X% faster convergence than Q-learning in deterministic environments by exploiting determinism through backward reward propagation. Applicable to robotics, game AI, and planning where environment dynamics are known/learnable.

### Acknowledgements
Thank Dr. Edward de Vere for early feedback on the backward propagation concept. Computing resources provided by the Fictional Institute of Reinforcement Learning (FIRL). Funded by grant #RL-2024-0042 from the Made-Up Science Foundation.

