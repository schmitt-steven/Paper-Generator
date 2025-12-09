### Experiment Plan: Testing RBQL vs. Standard Q-Learning in Deterministic Environments  

#### **Objective and Success Criteria**  
- **Objective**: Validate that Recursive Backwards Q-Learning (RBQL) converges to optimal policies significantly faster than standard Q-learning in deterministic episodic environments.
- **Success Criteria**:  
  - RBQL achieves a rolling 20-episode success rate of ≥0.9 in significantly fewer episodes than standard Q-learning.  
  - Statistical significance (p < 0.05) in episodes-to-convergence via independent t-test.  

#### **Current Implementation Details**  
- **Environment**: Deterministic Pong-like game.
  - **Randomized Start**: Initial ball X position is random [1-11], Velocity X is random [-1, 1]. This prevents trajectory memorization.
  - Terminal State: Ball reaches y=12. Reward +1 (win) or -1 (loss).
- **Hyperparameters**:  
  - $\gamma = 0.95$
  - **Epsilon decay**: **Per-episode**. `epsilon -= 1.0 / (400 * 0.8)`. This encourages strictly slower exploration which was found to differentiate the algorithms better in the randomized environment.
  - Max episodes: 400 per run.
  - Runs: 30 independent runs per algorithm.

#### **Algorithm Implementation**  
- **RBQL**: Stores transitions in a persistent model (never cleared). On terminal state, performs backward BFS update (exact Bellman with $\alpha=1$).
- **Standard Q-Learning**: Updates Q-table on every step ($\alpha=0.1$).

#### **Metrics & Output**  
1. **Convergence**: Episode where rolling 20-episode success rate first hits ≥0.9.
2. **Success Rate**: Calculated as proportion of wins (1.0 = win, 0.0 = loss).

#### **Required Plots**  
The script `rbql_vs_q_gemini.py` generates:
1.  **`comparison_plot.png`** (Learning Curve): Success Rate vs Episode.
2.  **`convergence_plot.png`** (Bar Chart): Mean episodes to convergence with error bars.