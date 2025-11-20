### Experimental Plan: Testing RBQL vs. Standard Q-Learning in Deterministic Sparse-Reward Environments

---

#### **Objective and Success Criteria**  
- **Objective**: Quantify the reduction in episodes required for convergence when using Recursive Backwards Q-Learning (RBQL) compared to standard Q-learning in a deterministic sparse-reward environment.  
- **Success Criteria**: RBQL achieves optimal policy in fewer episodes than Q-learning (α=1.0) on average across 50 trials, demonstrating the benefit of batch value iteration over online updates.

---

#### **Required Mathematical Formulas/Technical Details**  
- **Bellman Equation for Q-learning**:  
  $$
  Q(s, a) \leftarrow (1 - \alpha) \cdot Q(s, a) + \alpha \cdot [r + \gamma \cdot \max_{a'} Q(s', a')]
  $$  
- **RBQL Update Rule** (value iteration after episode completion):  
  $$
  Q(s, a) = r(s, a) + \gamma \cdot \max_{a'} Q(s', a')
  $$  
  Applied iteratively over all explored state-action pairs until convergence (max change < 1e-6).
- **Convergence Criterion**: Learned policy matches analytically computed optimal policy AND sufficient exploration achieved (all "go right" actions explored).
- **Exploration Policy**: ε-greedy (ε = 0.3) for all algorithms.
- **Initialization**: Optimistic initialization (Q = 1.0 for all state-action pairs) to encourage exploration.

---

#### **Experimental Setup**  
- **Environment**: 1D grid world (size N=15) with:  
  - Start state: `0`, Goal state: `14`.  
  - Actions: `left` (move to i-1 if i > 0) or `right` (move to i+1 if i < N-1).  
  - Rewards: `0` for all transitions except reaching goal (`+1`).  
  - Max steps per episode: 300 (prevents infinite episodes from random exploration).
- **Parameters**:  
  - Discount factor γ = 0.9.  
  - Standard Q-learning: α = 0.5 (moderate learning rate).  
  - Q-learning (α=1.0): Direct assignment for fair comparison with RBQL.
  - RBQL: Batch value iteration after each episode (effectively α = 1).  
- **Trials**: 50 independent runs per algorithm.  
- **Episode Limit**: Max 300 episodes per trial.  
- **Termination Condition**: Learned policy matches optimal policy (for Q-learning variants) OR sufficient exploration AND optimal policy (for RBQL).

---

#### **Metrics to Measure**  
- **Primary Metric**: Number of episodes required to achieve optimal policy (per trial).  
- **Secondary Metrics**:  
  - Average episodes across all trials.  
  - Standard deviation of episode counts (to assess consistency).  
- **Fair Comparison**: Q-learning (α=1.0) serves as baseline with same effective learning rate as RBQL.

---

#### **Implementation Approach**  
1. **Environment Class (`GridWorld`)**:  
   - Simulate 1D grid transitions and rewards.  
   - Track current state and episode termination (goal reached or max steps).  

2. **Optimal Q-Value Computation**:
   - Analytically compute ground truth Q-values by backward iteration from goal.
   - Used to verify policy optimality (argmax of learned Q matches argmax of optimal Q).

3. **Standard Q-Learning** (α=0.5 and α=1.0 variants):  
   - During each step in an episode:  
     ```python
     q_values[state][action] += alpha * (reward + gamma * np.max(q_values[next_state]) - q_values[state][action])
     ```  
   - Check policy optimality after each episode.

4. **RBQL Implementation**:  
   - `PersistentModel` to store all explored transitions.  
   - After episode ends, run value iteration until convergence:  
     ```python
     for _ in range(max_iterations):
         for state in explored_states:
             for action, next_state in transitions[state]:
                 q_values[state][action] = reward + gamma * np.max(q_values[next_state])
         if max_change < 1e-6:
             break
     ```  
   - Note: Topological sort cannot be used because grid has cycles (left/right transitions).

5. **Experiment Workflow**:  
   - For each trial (50 total):  
     1. Reset environment and Q-values (optimistic init = 1.0).  
     2. For each episode (max 300):  
        - Simulate agent until goal reached or step limit hit.  
        - Update Q-values (online for Q-learning, batch for RBQL).  
        - Check convergence criterion. If met, record episode count and stop trial.  
   - Repeat for all three algorithms independently.  

---

#### **Output Requirements**  
1. **JSON File (`results.json`)**:  
   ```json
   {
     "grid_size": 15,
     "trials": 50,
     "rbql_episodes": [4, 5, 5, ...],
     "standard_q_episodes": [12, 10, 14, ...],
     "q_alpha1_episodes": [6, 8, 5, ...],
     "rbql_avg": 4.6,
     "rbql_std": 0.5,
     "standard_q_avg": 11.3,
     "standard_q_std": 2.4,
     "q_alpha1_avg": 6.0,
     "q_alpha1_std": 2.3
   }
   ```  

2. **Stdout Summary**:  
   ```text
   RBQL:               4.6 ± 0.5 episodes
   Q-Learning (α=0.5): 11.3 ± 2.4 episodes
   Q-Learning (α=1.0): 6.0 ± 2.3 episodes
   
   RBQL vs Q-Learning (α=1.0): 1.30x faster
     ^ Fair comparison (same effective learning rate)
   
   RBQL vs Q-Learning (α=0.5): 2.46x faster
   ```  

3. **Plot**:  
   - Two-panel figure:
     - Left: Bar chart comparing average episodes with error bars (std dev).
     - Right: Box plot showing distribution of convergence times.
   - Title: "Convergence Comparison (15-State Grid, 50 trials)".  

---

#### **Runtime Optimization**  
- **Moderate Environment**: 1D grid (N=15) balances problem difficulty with computational efficiency.
- **Value Iteration**: Converges in <100 iterations per episode for this grid size.
- **50 Trials**: Provides statistically robust results (standard error ≈ std/√50).
- **Expected Runtime**: < 60 seconds in Python.

> **Note**: All code uses only `numpy`, `matplotlib`, and `seaborn`. Optimistic initialization ensures adequate exploration without requiring high ε values.