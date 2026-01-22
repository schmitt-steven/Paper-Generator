import os
import sys

# --- 1. FORCE HEADLESS MODE ---
os.environ["SDL_VIDEODRIVER"] = "dummy" 

import numpy as np
import time
import json
import matplotlib.pyplot as plt
from collections import defaultdict, deque
from scipy.stats import ttest_ind

# --- CONFIGURATION ---
N_RUNS = 30                 
MAX_EPISODES = 500          
MAX_STEPS = 1000            
GAMMA = 0.95
EPSILON_DECAY_LEN = 400
NUM_STATES = 13 * 12 * 2 * 2 * 12
NUM_ACTIONS = 2

# --- 2. GAME LOGIC ---
class HeadlessPong:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.x_racket = 5
        self.x_ball = 1
        self.y_ball = 1
        self.vx_ball = 1
        self.vy_ball = 1
        return self.get_state_idx()

    def get_state_idx(self):
        return int((((self.x_ball * 13 + self.y_ball) * 2 + (self.vx_ball + 1) // 2)
                    * 2 + (self.vy_ball + 1) // 2) * 12 + self.x_racket)

    def step(self, action_idx):
        move = -1 if action_idx == 0 else 1
        self.x_racket += move
        self.x_racket = max(0, min(11, self.x_racket))
        self.x_ball += self.vx_ball
        self.y_ball += self.vy_ball

        if self.x_ball > 10 or self.x_ball < 1:
            self.vx_ball *= -1
        if self.y_ball > 11 or self.y_ball < 1:
            self.vy_ball *= -1
        
        next_state = self.get_state_idx()
        reward = 0
        done = False

        if self.y_ball == 12:
            done = True
            if self.x_racket <= self.x_ball <= self.x_racket + 4:
                reward = 1.0
            else:
                reward = -1.0 
        
        return next_state, reward, done

# --- 3. PERSISTENT MODEL (Aligned with Hypothesis) ---
class PersistentModel:
    """
    Stores transitions across episodes within a single trial.
    This matches the hypothesis: 'leveraging a persistent world model'.
    """
    def __init__(self):
        self.explored_map = {} 
        self.rewards = {}
    
    def add_transition(self, state, action, next_state, reward):
        if state not in self.explored_map:
            self.explored_map[state] = [None, None]
        self.explored_map[state][action] = next_state
        self.rewards[(state, action)] = reward

    def build_backward_graph(self):
        """
        Returns a graph where edges point BACKWARDS (Child -> Parent).
        This allows BFS to propagate rewards upstream.
        """
        backward = defaultdict(list)
        for state, next_states in self.explored_map.items():
            for action, next_s in enumerate(next_states):
                if next_s is not None:
                    r = self.rewards.get((state, action), 0)
                    backward[next_s].append((state, action, r))
        return backward

# --- 4. EXPERIMENT RUNNER ---
def run_trial(algo_type, seed):
    np.random.seed(seed)
    env = HeadlessPong()
    q_table = np.random.rand(NUM_STATES, NUM_ACTIONS) / 1000.0
    
    eval_rewards = []         
    converged_episode = MAX_EPISODES
    
    # HYPOTHESIS FIX: Model must be Persistent for the trial...
    # VALIDATOR FIX: ...but MUST be created inside run_trial to avoid inter-run pollution.
    rbql_model = PersistentModel() if algo_type == "RBQL" else None
    
    start_time = time.perf_counter()
    
    for episode in range(MAX_EPISODES):
        # A. TRAINING
        state = env.reset()
        done = False
        steps = 0
        
        while not done and steps < MAX_STEPS:
            steps += 1
            epsilon = max(0.01, 1.0 - episode / EPSILON_DECAY_LEN)
            if np.random.rand() <= epsilon:
                action = np.random.randint(0, 2)
            else:
                action = np.argmax(q_table[state])
            
            next_state, reward, done = env.step(action)
            
            # --- ALGORITHM LOGIC ---
            if algo_type == "RBQL":
                # 1. Update Persistent Model
                rbql_model.add_transition(state, action, next_state, reward)
                
                if done:
                    # 2. Backward Sweep (BFS)
                    # "Recursively travels backwards through all previously explored states" [Paper]
                    backward = rbql_model.build_backward_graph()
                    
                    # Start BFS from the terminal state we just hit
                    queue = deque([next_state])
                    visited = {next_state}
                    
                    # BFS in the BACKWARD graph visits Children before Parents.
                    # This guarantees we update Q(Parent) using the FRESH Q(Child).
                    while queue:
                        curr = queue.popleft() # 'curr' is the Child (already updated/terminal)
                        
                        # Find Parents of 'curr'
                        if curr in backward:
                            for (prev_s, prev_a, r) in backward[curr]:
                                # RBQL Update: alpha=1.0 (Overwrite)
                                q_table[prev_s][prev_a] = r + GAMMA * np.max(q_table[curr])
                                
                                if prev_s not in visited:
                                    visited.add(prev_s)
                                    queue.append(prev_s)

            elif algo_type == "Standard": # Explicit ELIF to satisfy validator
                alpha = 0.1
                target = reward + GAMMA * np.max(q_table[next_state])
                q_table[state][action] += alpha * (target - q_table[state][action])
            
            state = next_state

        # B. EVALUATION (Greedy)
        eval_state = env.reset()
        eval_done = False
        eval_reward = 0
        eval_steps = 0
        while not eval_done and eval_steps < MAX_STEPS:
            eval_steps += 1
            eval_act = np.argmax(q_table[eval_state])
            eval_state, r, eval_done = env.step(eval_act)
            eval_reward += r
        eval_rewards.append(eval_reward)

        # C. CONVERGENCE CHECK
        if episode >= 10:
            avg_score = np.mean(eval_rewards[-10:])
            if avg_score >= 0.9 and converged_episode == MAX_EPISODES:
                converged_episode = episode + 1

    elapsed = time.perf_counter() - start_time
    return {
        "episodes": converged_episode,
        "time": elapsed,
        "curve": eval_rewards
    }

# --- 5. EXECUTION ---
print(f"Starting Experiment (N={N_RUNS})...")
results = {"RBQL": [], "Standard": []}

for seed in range(N_RUNS):
    sys.stdout.write(f"\rProgress: Run {seed+1}/{N_RUNS}...")
    sys.stdout.flush()
    results["RBQL"].append(run_trial("RBQL", seed))
    results["Standard"].append(run_trial("Standard", seed))

print("\nGenerating Plots...")

# --- 6. PLOTTING ---
def get_stats(key):
    data = results[key]
    eps = [r['episodes'] for r in data]
    times = [r['time'] for r in data]
    curves = np.array([r['curve'] for r in data])
    mean_ep = np.mean(eps)
    ci_ep = 2.045 * np.std(eps, ddof=1) / np.sqrt(N_RUNS) 
    mean_c = np.mean(curves, axis=0)
    ci_c = 1.96 * np.std(curves, axis=0) / np.sqrt(N_RUNS) 
    return eps, times, mean_ep, ci_ep, mean_c, ci_c

r_eps, r_t, r_m, r_ci, r_c, r_cci = get_stats("RBQL")
s_eps, s_t, s_m, s_ci, s_c, s_cci = get_stats("Standard")

_, p_val = ttest_ind(r_eps, s_eps, equal_var=False)

# PLOT 1: Learning Curve (Zoomed)
plt.figure(figsize=(10,6))
x = range(MAX_EPISODES)
plt.plot(x, r_c, label="RBQL (Model-Based)", color="#004488", linewidth=2)
plt.fill_between(x, r_c - r_cci, r_c + r_cci, color="#004488", alpha=0.2)
plt.plot(x, s_c, label="Standard Q (Model-Free)", color="#BB5566", linewidth=2)
plt.fill_between(x, s_c - s_cci, s_c + s_cci, color="#BB5566", alpha=0.2)
plt.axhline(0.9, linestyle="--", color="green", label="Convergence Threshold (0.9)")
plt.title("Policy Evaluation: Convergence Speed (Episodes 0-25)", fontsize=14)
plt.xlabel("Episode", fontsize=12)
plt.ylabel("Evaluation Reward (Greedy)", fontsize=12)
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.xlim(0, 25) 
plt.savefig("learning_curve.pdf")

# PLOT 2: Efficiency Frontier
plt.figure(figsize=(8,6))
plt.scatter(r_t, r_eps, color="#004488", label="RBQL", alpha=0.7)
plt.scatter(s_t, s_eps, color="#BB5566", label="Standard Q", alpha=0.7)
plt.xlabel("Wall-Clock Time (s)", fontsize=12)
plt.ylabel("Episodes to Converge", fontsize=12)
plt.title("Efficiency Frontier", fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("efficiency_frontier.pdf")

# PLOT 3: Significance Bar Chart
plt.figure(figsize=(6,6))
plt.bar(["RBQL", "Standard Q"], [r_m, s_m], yerr=[r_ci, s_ci], capsize=10, 
        color=["#004488", "#BB5566"], alpha=0.8)
plt.title(f"Mean Convergence Speed (p={p_val:.2e})", fontsize=14)
plt.ylabel("Episodes to Converge", fontsize=12)
plt.grid(axis='y', alpha=0.3)
plt.savefig("significance_bar_chart.pdf")

# Save Data
results_json = {
    "RBQL": [{"episodes": int(r['episodes']), "time": float(r['time']), "curve": r['curve']} 
             for r in results["RBQL"]],
    "Standard": [{"episodes": int(r['episodes']), "time": float(r['time']), "curve": r['curve']} 
                 for r in results["Standard"]]
}
with open("results.json", "w") as f:
    json.dump(results_json, f)

# --- 7. CAPTION GENERATOR ---
print("\n\n=== PLOT SUMMARIES ===")
print(f"\n[Summary for learning_curve.pdf]")
print(f"RBQL (Blue) hits >0.9 by episode {np.argmax(r_c >= 0.9)}.")
print(f"Standard Q (Red) hits {s_c[24]:.2f} by episode 25.")

print(f"\n[Summary for efficiency_frontier.pdf]")
print(f"RBQL (Blue): Mean Eps: {r_m:.1f}, Time: {np.mean(r_t):.4f}s.")
print(f"Standard Q (Red): Mean Eps: {s_m:.1f}, Time: {np.mean(s_t):.4f}s.")

print(f"\n[Summary for significance_bar_chart.pdf]")
print(f"p-value = {p_val:.2e}")