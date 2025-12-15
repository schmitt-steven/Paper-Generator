# Research Hypothesis

## Description
RBQL converges to optimal policies faster than standard Q-learning in deterministic, episodic environments by leveraging a persistent world model and backward reward propagation, eliminating the need for repeated visits to update Q-values.

## Rationale
Standard Q-learning suffers from extreme sample inefficiency in deterministic, episodic environments because each transition update relies on incremental TD updates (α < 1) and requires multiple visits to the same state-action pair to propagate reward signals from terminal states. RBQL addresses this by leveraging deterministic environment dynamics to propagate terminal rewards backward through a persistent state-transition model that accumulates across episodes, updating all known states in a single pass upon reaching a terminal state. This approach directly exploits the deterministic structure of the environment, which is systematically underutilized in standard Q-learning.

## Success Criteria
RBQL demonstrates faster convergence to optimal policies compared to standard Q-learning in deterministic, episodic environments as evidenced by a learning curve showing higher cumulative reward per episode and fewer episodes required to reach optimal performance.
