# Experimental Plan: RBQL Convergence Stability vs. Q-Learning

## Objective & Success Criteria

**Objective:**  
Evaluate whether Recursive Backwards Q-Learning (RBQL) demonstrates superior convergence stability compared to standard Q-learning when solving episodic tasks with sparse rewards.

**Success Criteria:**  
- RBQL achieves faster and more stable learning curves
- Reduced variance in final performance across multiple runs
- Less sensitivity to hyperparameter choices (e.g., learning rate α, exploration ε)
- Consistent convergence to optimal policy within limited episodes

---

## Mathematical Formulation & Technical Details

### Bellman Update Equations:
Standard Q-learning:
$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha [r_t + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t)]$$

RBQL with α = 1 (direct assignment):
$$Q(s_t, a_t) \leftarrow r_t + \gamma \max_{a'} Q(s_{t+1}, a')$$

### Key Metrics:
- **Episode Return**: Total reward per episode
- **Convergence Time**: Number of episodes to reach 95% of optimal return
- **Variance in Final Returns**: Standard deviation across runs
- **Exploration Efficiency**: Average number of unique states visited per episode

---

## Experimental Setup

### Environment:
- **Task**: Grid-world navigation with sparse rewards (e.g., terminal reward only at goal)
- **Grid Size**: 10x10 deterministic grid (small enough for fast computation)
- **Actions**: Up, Down, Left, Right (4 actions)
- **Reward Structure**: 
  - +10 for reaching terminal state
  - -1 for each step taken (sparse reward)
- **Episode Length**: Max 50 steps

### Algorithms Compared:
1. **Q-Learning** (α = 0.1, ε-greedy)
2. **RBQL** (α = 1, ε-greedy)

### Hyperparameters:
| Parameter | Value |
|----------|-------|
| γ (discount) | 0.95 |
| ε (exploration) | 0.1 |
| α (learning rate) | 0.1 for Q-learning, 1.0 for RBQL |
| Max Episodes | 200 |
| Runs | 10 independent runs |

### Evaluation Strategy:
- Run each algorithm 10 times independently
- Record episode returns after every 5 episodes
- Compute average performance over time
- Analyze convergence stability and variance

---

## Metrics to Measure

1. **Average Episode Return**: Mean reward per episode across all runs
2. **Convergence Time**: Number of episodes required to achieve ≥95% of optimal return
3. **Standard Deviation of Final Returns**: Variance in final performance
4. **Exploration Efficiency**: Average number of unique states visited per episode
5. **Policy Consistency**: Percentage of consistent optimal actions chosen across runs

---

## Implementation Approach

### Algorithm Design:
1. **Persistent Model Storage**:
   - Maintain `explored_map` and `rewards`
   - Build backward graph at episode end using BFS

2. **Training Loop**:
   ```python
   for episode in range(max_episodes):
       state = reset_env()
       while not is_terminal(state):
           action = ε_greedy(Q, state)
           next_state, reward = step(state, action)
           model.add_transition(state, action, next_state, reward)
           state = next_state

       # Backward propagation after episode
       propagate_reward_rbql(state, Q, model)
   ```

3. **Evaluation**:
   - Store returns for each episode
   - Calculate mean/stddev across runs
   - Generate plots of average return vs episodes

---

## Output Requirements

### 1. JSON File (`results.json`):
```json
{
  "algorithm": "RBQL",
  "metrics": {
    "average_return_per_episode": [0, 2, 4, ...],
    "convergence_time": 45,
    "final_return_stddev": 0.8,
    "avg_unique_states_per_episode": 32.5
  },
  "run_details": [
    {"run_id": 1, "returns": [0, 2, 4, ...]},
    ...
  ]
}
```

### 2. Console Output:
```
RBQL Performance Summary:
- Average Return per Episode: 8.7 ± 1.2
- Convergence Time (95% optimal): 42 episodes
- Final Return Variance: 0.9
- Avg Unique States Visited: 34.2

Q-Learning Performance Summary:
- Average Return per Episode: 6.3 ± 2.1
- Convergence Time (95% optimal): 87 episodes
- Final Return Variance: 2.4
- Avg Unique States Visited: 30.1

Conclusion: RBQL converges faster and more stably than Q-learning.
```

### 3. Visualization:
Plot showing average episode return over time for both algorithms (with error bands), highlighting convergence differences.

---

## Execution Constraints & Optimization

To ensure completion under 5 minutes:

- **Grid Size**: Reduced to 10×10
- **Episodes**: Limited to 200 max
- **Runs**: Only 10 independent runs
- **Sampling Frequency**: Record every 5 episodes
- **Optimization**: Precompute reward structure; use efficient BFS queue

This setup allows full simulation of both algorithms while maintaining statistical significance and meaningful comparison within time constraints.