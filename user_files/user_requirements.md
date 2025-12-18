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
4. Describe experiment setup: environment (simple grid or pong-like game), state space, action space, hyperparameters (γ, ε decay schedule), baseline (standard Q-learning with same ε schedule)

peudo code

### Results
Compare RBQL vs standard Q-learning on:
- Episodes to convergence (optimal policy)
- Cumulative reward over episodes
- Include statistical measures (like mean, std over multiple runs)

Try to get generalizable results.

Required plots:
1. **Learning curve**: Cumulative reward (y-axis) vs Episode number (x-axis), two lines (RBQL vs Q-learning), with shaded std regions
2. **Convergence speed**: Bar chart showing episodes required to reach 90% of optimal performance for each algorithm

### Discussion
Analyze why RBQL outperforms Q-learning in deterministic settings. Discuss limitations: only works for deterministic environments, requires storing full transition model (memory), episodic tasks only. Suggest extensions: stochastic environments (weighted propagation), continuous state spaces, memory-efficient model compression.

### Conclusion
2-3 sentences: RBQL demonstrates X% faster convergence than Q-learning in deterministic environments by exploiting determinism through backward reward propagation. Applicable to robotics, game AI, and planning where environment dynamics are known/learnable.

### Acknowledgements
Thank Dr. Edward de Vere for early feedback on the backward propagation concept. Computing resources provided by the Fictional Institute of Reinforcement Learning (FIRL). Funded by grant #RL-2024-0042 from the Made-Up Science Foundation.

