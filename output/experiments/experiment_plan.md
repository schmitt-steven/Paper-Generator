**Experiment Plan: Testing RBQL vs Standard Q-Learning in Deterministic Episodic Environments**

---

**Objective and Success Criteria:**  
The objective is to empirically validate that Recursive Backwards Q-Learning (RBQL) converges to optimal policies faster than standard Q-learning in deterministic, episodic environments. Success is defined as RBQL achieving an average cumulative reward ≥ 0.9 over a rolling window of 10 episodes in significantly fewer episodes and less wall-clock time than standard Q-learning, with statistical significance (p < 0.05) confirmed via Welch’s t-test.

---

**Required Mathematical Formulas/Technical Details:**  
- **Bellman Optimality Update (RBQL):**  
  Upon reaching a terminal state, update all known Q-values in reverse topological order via BFS:  
  `Q(s,a) = R(s,a) + γ * max(Q(s'))` with α=1 (no incremental averaging).  
- **Standard Q-Learning Update:**  
  `Q(s,a) ← (1−α) * Q(s,a) + α * [R(s,a) + γ * max(Q(s'))]` with α=0.1 (fixed).  
- **Convergence Criterion:**  
  First episode where rolling average of last 10 rewards ≥ 0.9.  
- **Confidence Intervals:**  
  95% CI for mean episodes and wall-clock time computed as: `mean ± t*(s/√n)` where t is critical value from t-distribution (df = n−1).  
- **Welch’s t-test:**  
  Used to compare means between RBQL and standard Q-learning (unequal variances, unequal sample sizes).  

---

**Experiment Setup:**  
- **Environment:** Deterministic Breakout-like game (as in provided code) with discrete state space (`num_of_states = 13*12*2*2*12`), 2 actions (left/right).  
- **Algorithms Compared:**  
  - RBQL: Uses persistent transition model, backward BFS update on terminal state with α=1.  
  - Standard Q-learning: Same state/action space, fixed α=0.1, no model storage or backward propagation.  
- **Runs:** 30 independent runs per algorithm (seeds 0–29).  
- **Episode Limit:** Max 500 episodes per run.  
- **Termination Condition:** Convergence when rolling average of last 10 rewards ≥ 0.9 (or episode limit reached).  
- **Exploration:** Epsilon-greedy with decay: `epsilon = max(0, 1.0 - episode/400)` (same for both).  
- **Hardware:** Single-threaded CPU execution; no GPU used. Wall-clock time measured via `time.time()`.  
- **Randomization:** Each run uses a unique seed for action selection and environment initialization.  

---

**Metrics to Measure:**  
1. **Episodes to Convergence:** Number of episodes until rolling average reward ≥ 0.9.  
2. **Wall-clock Time to Convergence:** Total seconds elapsed until convergence (rendering MUST be disabled).  
3. **Final Average Reward:** Mean reward over last 10 episodes (for robustness check).  
4. **Statistical Significance:** Welch’s t-test p-value for both metrics.  
5. **95% Confidence Intervals:** For mean episodes and time across 30 runs.  

---

**Implementation Approach:**  
1. **Modify Existing Code to Support Comparison:**  
   - Refactor `recursive_backwards_q_learning.py` into a reusable function that accepts an algorithm flag (`"RBQL"` or `"StandardQ"`).  
   - Remove PyGame rendering to reduce overhead (keep only for debugging if needed; disable during runs).  
   - Replace `file.write()` with structured logging of episode rewards.
   - Crucial: The program must run in "Headless Mode" (no window, no drawing) to keep the execution time low.
2. **Standard Q-Learning Implementation:** 
   - Remove `PersistentModel` and `propagate_reward_rbql`.  
   - Replace backward update with standard Q-update: `q_values[state][action_index] += 0.1 * (reward + gamma * np.max(q_values[next_state]) - q_values[state][action_index])`.  
3. **Run Controller:**  
   - Loop over 60 total runs (30 per algorithm), each with unique seed.  
   - For each run:  
     - Initialize Q-table (same random initialization per seed).  
     - Simulate episodes until convergence or 500 episodes.  
     - Record: episode count at convergence, wall-clock time, final 10-episode avg reward.  
4. **Data Collection:**  
   - Store results in JSON: `{"RBQL": [{"episodes": x, "time": y, "final_avg_reward": z}, ...], "StandardQ": [...]}`.  
5. **Analysis & Visualization:**  
   - Compute means, 95% CIs, and Welch’s t-test p-values.  
   - Generate three plots:  
     a) **Learning Curve:** Episode vs. mean reward (shaded 95% CI for each algorithm).  
     b) **Efficiency Frontier:** Scatter plot of wall-clock time (x) vs. episodes to converge (y), with 60 points (30 per algorithm).  
     c) **Significance Bar Chart:** Mean episodes to converge with 95% CI error bars and annotated p-value.  
   - Save all plots as `.pdf`.  
6. **Output:**  
   - Print to stdout: concise summary of mean episodes, time, p-values, and conclusion.  
   - Ensure total runtime < 5 minutes (disable rendering, reduce max episodes if needed).  

---

**Output Requirements:**  
- **JSON File (`results.json`):** Structured with keys `"RBQL"` and `"StandardQ"`, each containing list of dicts: `{"episodes": int, "time": float, "final_avg_reward": float}`.  
- **Stdout Output:** One-line summary: e.g., “RBQL converged in 82.3±5.1 episodes vs StandardQ’s 247.6±18.9 (p=0.001). Time: 4.2s vs 12.8s.”  
- **Plots (saved as .pdf):**  
  - `learning_curve.pdf`  
  - `efficiency_frontier.pdf`  
  - `significance_bar_chart.pdf`  

---

**Constraints & Optimization for Speed:**  
- Disable PyGame rendering (`pyg.display.set_mode()` skipped) to reduce per-episode overhead.  
- Use `np.random.seed(seed)` at start of each run for reproducibility.  
- Limit max episodes to 500 (sufficient for convergence in deterministic setting).  
- Use vectorized operations where possible; avoid dictionary lookups in inner loops.  
- Precompute state indices via `getState()` without object creation overhead.  
- Run all 60 experiments sequentially in a single process (no multiprocessing).  

---