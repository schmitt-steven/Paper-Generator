**Experiment Plan: Testing RBQL vs Standard Q-Learning in Deterministic Episodic Environments**

---

**Objective and Success Criteria:**  
The objective is to empirically validate that Recursive Backwards Q-Learning (RBQL) converges to optimal policies significantly faster than standard Q-learning in deterministic, episodic environments. Success is defined as:  
- RBQL achieving 90% of optimal cumulative reward in fewer episodes than standard Q-learning.  
- RBQL’s learning curve showing steeper initial improvement and lower variance in convergence time across multiple runs.  
- Statistical evidence (mean ± std) demonstrating superior sample efficiency of RBQL.

---

**Required Mathematical Formulas/Technical Details:**  
- **Bellman Optimality Update (RBQL):**  
  Upon reaching a terminal state, for each (s, a) in the backward BFS order:  
  `Q(s, a) ← R(s, a) + γ · max_{a'} Q(s', a')` with α = 1 (no incremental averaging).  
- **Standard Q-learning Update:**  
  `Q(s, a) ← Q(s, a) + α · [R(s, a) + γ · max_{a'} Q(s', a') - Q(s, a)]` with α = 0.1 (fixed).  
- **Epsilon-Greedy Exploration:**  
  `ε = max(0.05, 1.0 - episode / 400)` for both algorithms (identical decay schedule).  
- **State Space:** Discrete, deterministic mapping from continuous game state (ball x,y,vx,vy; racket position) to integer index (13×12×2×2×12 = 6,240 states).  
- **Action Space:** {Left (-1), Right (+1)} → 2 discrete actions.  
- **Terminal Condition:** Ball reaches bottom row (y=12); reward = +1 if racket overlaps ball, else -1.  
- **Optimal Policy Benchmark:** Max achievable reward per episode = +1; optimal cumulative reward over N episodes = N (if always successful).

---

**Experiment Setup:**  
- **Environment:** Modified Pong-like game from provided code (240×260 pixels, discrete state mapping via `getState()`).  
- **Episodes:** 300 total episodes per run (sufficient for convergence; under 5 min runtime).  
- **Runs:** 10 independent runs per algorithm to compute mean and std.  
- **Hyperparameters (shared):**  
  - γ = 0.95  
  - ε decay: `ε = max(0.05, 1.0 - episode / 400)`  
  - State space: 6,240 discrete states; 2 actions.  
- **Baseline:** Standard Q-learning with α = 0.1, no persistent model, single-step TD updates.  
- **RBQL:** Uses persistent transition model; backward BFS update on terminal state with α=1.  
- **Initialization:** Both algorithms initialize Q-values to small random values (uniform [0, 0.01]).  
- **Termination:** Episode ends when ball reaches y=12; next episode resets game state.  

---

**Metrics to Measure:**  
1. **Cumulative Reward per Episode**: Running sum of rewards within each episode (used for learning curve).  
2. **Episodes to Convergence**: First episode where cumulative reward ≥ 90% of theoretical maximum (i.e., ≥ 0.9 per episode on average over last 5 episodes).  
3. **Mean and Standard Deviation** of convergence episodes across 10 runs for each algorithm.  
4. **Final Policy Quality**: Average cumulative reward over last 10 episodes (to assess asymptotic performance).  

---

**Implementation Approach:**  
- **Modify existing code to support both algorithms in a single run loop.**  
  - Introduce `algorithm_mode = "RBQL"` or `"StandardQ"` as a switch.  
  - For Standard Q-learning: replace `propagate_reward_rbql()` with single-step Q-update using α=0.1.  
  - For RBQL: retain existing backward BFS update logic.  
- **Remove Pygame rendering** to speed up execution (no visualization needed).  
- **Store episode rewards and convergence metrics per run.**  
- **Run 10 independent trials** (each with separate Q-table and model initialization).  
- **For each run:**  
  - Initialize Q-tables and (for RBQL) persistent model.  
  - For episode in 1 to 300:  
    - Simulate one episode using current policy (epsilon-greedy).  
    - Record total reward for the episode.  
    - Update Q-values according to algorithm mode.  
  - Record: total episodes until convergence (if reached), final 10-episode avg reward.  
- **Post-experiment:** Compute mean/std of convergence episodes and final rewards across runs.

---

**Output Requirements:**  
- **JSON File (`results.json`):**  
  ```json
  {
    "RBQL": {
      "convergence_episodes": [23, 18, 25, ...],   // 10 values
      "cumulative_rewards": [[r1_1, r1_2, ...], [r2_1, r2_2, ...]], // 10 runs × 300 episodes
      "final_avg_reward": 0.94,
      "mean_convergence": 21.5,
      "std_convergence": 3.2
    },
    "StandardQ": {
      "convergence_episodes": [145, 160, ...],
      "cumulative_rewards": [...],
      "final_avg_reward": 0.82,
      "mean_convergence": 152.3,
      "std_convergence": 18.7
    }
  }
  ```  
- **Stdout Output (concise):**  
  ```
  RBQL: Mean episodes to convergence = 21.5 ± 3.2
  StandardQ: Mean episodes to convergence = 152.3 ± 18.7
  RBQL is 6.9x faster to converge.
  Final reward: RBQL=0.94, StandardQ=0.82
  Hypothesis supported: RBQL converges significantly faster.
  ```  
- **Plots (saved as .pdf):**  
  1. **Learning Curve**: Episode (x) vs Cumulative Reward (y), with two lines (RBQL, StandardQ) and shaded ±1 std regions.  
  2. **Convergence Speed Bar Chart**: Two bars (RBQL, StandardQ) showing mean episodes to 90% optimal performance with error bars (±std).  

---

**Execution Constraints:**  
- Total runtime ≤ 5 minutes: Achieved by disabling rendering, limiting to 300 episodes × 10 runs = 3,000 episodes.  
- State space is fixed and small (6,240 states); BFS backward propagation is efficient due to limited branching.  
- No parallelization; single-threaded but optimized for speed (no rendering, no I/O during episode).  
- All computations use NumPy arrays; no heavy matrix operations.  

--- 

**Adaptations to Provided Code:**  
- Remove `pygame` rendering and display logic.  
- Replace main game loop with a pure simulation loop over episodes and runs.  
- Add `algorithm_mode` switch to toggle between RBQL and Standard Q-learning updates.  
- Store rewards per episode per run in nested lists for later analysis.  
- Add JSON serialization and matplotlib plotting at end of all runs.  
- Ensure Q-tables are re-initialized for each run to avoid carryover effects.