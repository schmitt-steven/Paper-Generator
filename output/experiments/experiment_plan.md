# Experiment Plan: Recursive Backwards Q-Learning (RBQL) vs. Prioritized Experience Replay

## 1. Objective and Success Criteria
- **Objective:** Empirically validate the hypothesis that Recursive Backwards Q-Learning (RBQL) achieves superior sample efficiency compared to Prioritized Experience Replay (PER) in deterministic episodic environments with sparse rewards.
- **Success Criteria:** 
  - RBQL reaches a target cumulative reward threshold (e.g., >0.5 average episode return) in significantly fewer episodes than PER across multiple random seeds.
  - RBQL demonstrates faster wall-clock convergence time despite the overhead of backward graph traversal.
  - Statistical significance (p < 0.05) is observed in the episode count required for convergence between methods.

## 2. Required Mathematical Formulas & Technical Details
- **Bellman Optimality (RBQL):** $Q(s, a) \leftarrow R_{propagated}$ where $R_{propagated} = r \times \gamma$ is propagated backwards via BFS with direct assignment ($\alpha = 1$). A persistent `Q_updated` mask ensures each state-action pair is assigned at most once.
- **Bellman Update (PER):** $Q(s, a) \leftarrow Q(s, a) + \alpha [r + \gamma \max_{a'} Q(s', a') - Q(s, a)]$.
- **State Space:** Discrete integer encoding based on ball/racket coordinates and velocities (7488 states).
- **Environment:** Deterministic Pong-like dynamics. No stochasticity in transitions; only exploration noise ($\epsilon$-greedy).

## 3. Experiment Setup
- **Environment:** Shared `get_state` / `step_env` / `reset_env` helper functions encapsulating the Pong-like dynamics. Headless execution (SDL dummy drivers, Agg matplotlib backend).
- **Algorithms:** 
  - **Method A (RBQL):** `run_backward_prop` — BFS backward propagation from terminal states with direct Q-value assignment.
  - **Method B (Baseline):** `run_prioritized_experience_replay` — standard Q-learning with prioritized replay buffer.
- **Hyperparameters:** 
  - Discount Factor ($\gamma$): 0.95
  - Learning Rate ($\alpha$): 0.1 (PER), 1.0 effective (RBQL direct assignment)
  - $\epsilon$-Decay: Linear decay from 1.0 to 0 over 400 steps.
  - Buffer Size (PER): 1000, Batch Size: 32, PER $\alpha$: 0.6, PER $\epsilon$: 1e-4.
  - Q-table initialization: Small random values (`rand / 1000`).
- **Runs:** 30 independent seeds per algorithm (seeds 0–29 for RBQL, 1000–1029 for PER).
- **Budget:** Maximum 300 episodes per run.

## 4. Metrics to Measure
1.  **Sample Complexity:** Number of episodes required to reach a stable average return (moving average $\geq 0.5$ over a window of 20, sustained until end).
2.  **Cumulative Reward:** Mean $\pm$ std of episode rewards across seeds (plotting curve).
3.  **Wall-Clock Time:** Total execution time per algorithm across all seeds + scaling across episode budgets.
4.  **Total Reward Sum:** Sum of all rewards per run (distribution via boxplot).

## 5. Implementation Approach
- **Code Structure:** Single self-contained script `experiment.py`.
  - Environment dynamics factored into `get_state`, `step_env`, `reset_env`.
  - Algorithm implementations: `run_backward_prop`, `run_prioritized_experience_replay` (with helper `update_q_per`).
  - `run_experiment()` orchestrates seeding, timing, aggregation, and plotting.
  - Environment variables set for headless execution (`SDL_VIDEODRIVER='dummy'`, matplotlib `Agg` backend).
- **Main Experiment:** Loop over 30 seeds; for each seed, run both algorithms and record per-episode rewards and wall-clock time.
- **Scaling Experiment:** Run both algorithms at episode budgets [50, 100, 150, 200, 250, 300] with 5 seeds each, recording mean wall-clock time per budget.
- **Convergence Detection:** For each seed, find the first episode where the moving average (window=20) reaches $\geq 0.5$ and remains above it for all subsequent windows. Return 300 (max episodes) if no convergence detected.
- **Aggregation:** Compute mean and std of convergence episodes, total reward sums, and wall-clock times across seeds.

## 6. Output Requirements
- **Stdout:** 
  - Print the **Hyperparameter Settings Table** (Markdown format).
  - Print the **Pseudocode Algorithm Block** for RBQL (10-20 lines, summarizing the backward pass logic).
  - Print **Convergence Metrics Table**: Mean $\pm$ Std for convergence episode, total reward sum, and wall-clock time for both methods.
  - Print **Scaling Experiment** results per episode budget.
- **Visualizations (PDF):**
  1. `sample_complexity.pdf`: Line plot of Mean Reward vs. Episodes (RBQL vs PER) with shaded $\pm 1$ std bands.
  2. `boxplots.pdf`: Side-by-side boxplots for Total Reward Sum and Convergence Episode distributions.
  3. `wall_clock_time.pdf`: Bar chart comparing mean wall-clock time between algorithms.
  4. `wall_clock_scaling.pdf`: Line graph showing computation time vs. episode budget for both algorithms.
  - **Style:** Seaborn `ticks` theme with `colorblind` palette, `sns.despine()`. Distinct markers and line styles for monochrome readability. Hatching on PER elements for print distinction.
  - **Constraint:** Use `plt.savefig(..., format='pdf')`. No `plt.show()`. Close all figures after saving.
- **JSON:** `results.json` containing hyperparameters, per-algorithm metrics (mean/std time, convergence episodes per seed, total rewards per seed), and scaling experiment data.
- **Performance:** Total execution time should stay under 10 minutes.

## 7. Safety & Headless Execution
- **No GUI:** Explicitly set `os.environ['SDL_VIDEODRIVER'] = 'dummy'` and `os.environ['SDL_AUDIODRIVER'] = 'dummy'` at the start of the script.
- **No Blocking:** All plotting uses `savefig` and closes figures (`plt.close()`) to prevent memory leaks.
- **Determinism:** Fix `np.random.seed(seed)` at the start of each run for reproducibility.

## 8. Required Artifacts for Paper (to be generated in output)
- **Pseudocode:** A text block describing the RBQL backward traversal (State -> Reverse Map -> BFS Queue -> Update).
- **Hyperparameter Table:** A markdown table listing $\gamma, \alpha, \epsilon_{start}, \epsilon_{end}$, Buffer Size, PER parameters.
- **Convergence Table:** A markdown table with columns: Method, Mean Episodes to Converge, Std Dev, Mean Total Reward, Mean Time (s).
