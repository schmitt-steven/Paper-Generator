import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import deque, defaultdict
import json
import os

class PersistentModel:
    def __init__(self):
        self.explored_map = {}
        self.rewards = {}

    def add_transition(self, state, action_index, next_state, reward):
        if state not in self.explored_map:
            self.explored_map[state] = [None, None]
        self.explored_map[state][action_index] = next_state
        self.rewards[(state, action_index)] = reward

    def get_next_state(self, state, action_index):
        if state not in self.explored_map:
            return None
        return self.explored_map[state][action_index]

    def get_reward(self, state, action_index):
        return self.rewards.get((state, action_index), 0)

    def build_topological_order(self):
        graph = defaultdict(list)
        in_degree = defaultdict(int)
        all_states = set()

        for state, next_states in self.explored_map.items():
            all_states.add(state)
            for action_index, ns in enumerate(next_states):
                if ns is not None:
                    all_states.add(ns)
                    graph[ns].append(state)
                    in_degree[state] += 1

        queue = deque([s for s in all_states if in_degree.get(s, 0) == 0])
        topo_order = []
        
        while queue:
            current = queue.popleft()
            topo_order.append(current)
            for neighbor in graph[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
                    
        return topo_order


class GridWorld:
    def __init__(self, size=10):
        self.size = size
        self.goal = size - 1

    def reset(self):
        self.state = 0
        return self.state

    def step(self, action):
        if action == 0:
            next_state = max(0, self.state - 1)
        else:
            next_state = min(self.size - 1, self.state + 1)
        reward = 1 if next_state == self.goal else 0
        self.state = next_state
        return next_state, reward


def propagate_reward_rbql(model, q_values, gamma=0.9):
    """Value iteration through explored transitions until convergence."""
    # Can't use topological sort - grid has cycles (left/right between states)
    # Use value iteration instead
    for _ in range(100):  # Max iterations
        old_q = q_values.copy()
        for state in model.explored_map:
            for action_index, next_state in enumerate(model.explored_map[state]):
                if next_state is None:
                    continue
                reward = model.get_reward(state, action_index)
                max_next_q = np.max(q_values[next_state])
                q_values[state][action_index] = reward + gamma * max_next_q
        
        # Check convergence
        if np.max(np.abs(q_values - old_q)) < 1e-6:
            break


def compute_optimal_q(size, gamma):
    """Compute ground truth optimal Q-values for the grid."""
    q_optimal = np.zeros((size, 2))
    goal = size - 1
    
    for state in range(goal - 1, -1, -1):
        next_left = max(0, state - 1)
        reward_left = 1 if next_left == goal else 0
        q_optimal[state][0] = reward_left + gamma * np.max(q_optimal[next_left])
        
        next_right = min(size - 1, state + 1)
        reward_right = 1 if next_right == goal else 0
        q_optimal[state][1] = reward_right + gamma * np.max(q_optimal[next_right])
    
    return q_optimal


def check_policy_optimal(q_values, q_optimal):
    """Check if learned policy matches optimal policy."""
    learned_policy = np.argmax(q_values, axis=1)
    optimal_policy = np.argmax(q_optimal, axis=1)
    return np.all(learned_policy == optimal_policy)


def check_sufficient_exploration(model, size):
    """Check if enough state-action pairs explored for optimal policy."""
    for state in range(size - 1):
        if state not in model.explored_map:
            return False
        if model.explored_map[state][1] is None:
            return False
    return True


def run_standard_q_learning(size, gamma, epsilon, alpha, max_episodes):
    """Standard Q-learning with proper convergence check."""
    q_values = np.ones((size, 2))  # Optimistic initialization
    q_optimal = compute_optimal_q(size, gamma)
    env = GridWorld(size=size)
    max_steps = size * 20
    
    for episode in range(max_episodes):
        state = env.reset()
        done = False
        steps = 0
        
        while not done and steps < max_steps:
            if np.random.rand() < epsilon:
                action = np.random.randint(2)
            else:
                action = np.argmax(q_values[state])

            next_state, reward = env.step(action)
            best_next_q = np.max(q_values[next_state])
            q_values[state][action] += alpha * (reward + gamma * best_next_q - q_values[state][action])
            state = next_state
            steps += 1
            if state == env.goal:
                done = True

        if check_policy_optimal(q_values, q_optimal):
            return episode + 1
    
    return max_episodes


def run_rbql(size, gamma, epsilon, max_episodes):
    """RBQL with proper convergence check."""
    q_values = np.ones((size, 2))  # Optimistic initialization
    q_optimal = compute_optimal_q(size, gamma)
    model = PersistentModel()
    env = GridWorld(size=size)
    max_steps = size * 20
    
    for episode in range(max_episodes):
        state = env.reset()
        done = False
        steps = 0
        
        while not done and steps < max_steps:
            if np.random.rand() < epsilon:
                action = np.random.randint(2)
            else:
                action = np.argmax(q_values[state])

            next_state, reward = env.step(action)
            model.add_transition(state, action, next_state, reward)
            state = next_state
            steps += 1
            if state == env.goal:
                done = True

        propagate_reward_rbql(model, q_values, gamma)

        if check_sufficient_exploration(model, size) and check_policy_optimal(q_values, q_optimal):
            return episode + 1
    
    return max_episodes


def run_q_learning_alpha1(size, gamma, epsilon, max_episodes):
    """Q-learning with alpha=1 for fair comparison with RBQL."""
    q_values = np.ones((size, 2))  # Optimistic initialization
    q_optimal = compute_optimal_q(size, gamma)
    env = GridWorld(size=size)
    max_steps = size * 20
    
    for episode in range(max_episodes):
        state = env.reset()
        done = False
        steps = 0
        
        while not done and steps < max_steps:
            if np.random.rand() < epsilon:
                action = np.random.randint(2)
            else:
                action = np.argmax(q_values[state])

            next_state, reward = env.step(action)
            best_next_q = np.max(q_values[next_state])
            q_values[state][action] = reward + gamma * best_next_q
            state = next_state
            steps += 1
            if state == env.goal:
                done = True

        if check_policy_optimal(q_values, q_optimal):
            return episode + 1
    
    return max_episodes


# Experiment parameters
N = 15  # Larger grid - more signal
gamma = 0.9
epsilon = 0.3  # Lower epsilon - optimistic init drives exploration
alpha_standard = 0.5
max_episodes_per_trial = 300

trials = 50  # More trials for tighter confidence intervals

rbql_episodes = []
standard_q_episodes = []
q_alpha1_episodes = []

print(f"Running {trials} trials on {N}-state grid...")
print(f"Parameters: gamma={gamma}, epsilon={epsilon}, alpha_standard={alpha_standard}")
print()
print(f"Parameters: gamma={gamma}, epsilon={epsilon}, alpha_standard={alpha_standard}")
print()

for trial in range(trials):
    rbql_ep = run_rbql(N, gamma, epsilon, max_episodes_per_trial)
    std_ep = run_standard_q_learning(N, gamma, epsilon, alpha_standard, max_episodes_per_trial)
    alpha1_ep = run_q_learning_alpha1(N, gamma, epsilon, max_episodes_per_trial)
    
    rbql_episodes.append(rbql_ep)
    standard_q_episodes.append(std_ep)
    q_alpha1_episodes.append(alpha1_ep)
    
    if (trial + 1) % 10 == 0:
        print(f"  Completed {trial + 1}/{trials} trials")

results = {
    "grid_size": N,
    "trials": trials,
    "rbql_episodes": rbql_episodes,
    "standard_q_episodes": standard_q_episodes,
    "q_alpha1_episodes": q_alpha1_episodes,
    "rbql_avg": float(np.mean(rbql_episodes)),
    "rbql_std": float(np.std(rbql_episodes)),
    "standard_q_avg": float(np.mean(standard_q_episodes)),
    "standard_q_std": float(np.std(standard_q_episodes)),
    "q_alpha1_avg": float(np.mean(q_alpha1_episodes)),
    "q_alpha1_std": float(np.std(q_alpha1_episodes)),
}

os.makedirs('plots', exist_ok=True)
with open('results.json', 'w') as f:
    json.dump(results, f, indent=2)

# Plot results
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

algorithms = ['RBQL', 'Q-Learning\n(α=0.5)', 'Q-Learning\n(α=1.0)']
avg_episodes = [results['rbql_avg'], results['standard_q_avg'], results['q_alpha1_avg']]
std_devs = [results['rbql_std'], results['standard_q_std'], results['q_alpha1_std']]

colors = sns.color_palette("Set2", 3)
bars = axes[0].bar(algorithms, avg_episodes, color=colors)
axes[0].errorbar(np.arange(len(algorithms)), avg_episodes, yerr=std_devs, 
                  fmt='none', capsize=5, color='black')
axes[0].set_ylabel('Episodes to Optimal Policy')
axes[0].set_title(f'Convergence Comparison ({N}-State Grid, {trials} trials)')

for bar, avg, std in zip(bars, avg_episodes, std_devs):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 1,
                  f'{avg:.1f}±{std:.1f}', ha='center', va='bottom', fontsize=9)

data_for_box = [rbql_episodes, standard_q_episodes, q_alpha1_episodes]
bp = axes[1].boxplot(data_for_box, tick_labels=algorithms, patch_artist=True)
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
axes[1].set_ylabel('Episodes to Optimal Policy')
axes[1].set_title('Distribution of Convergence Times')

plt.tight_layout()
plt.savefig('plots/convergence_comparison.png', dpi=150)
print("\nPlot saved to plots/convergence_comparison.png")

# Print results
print("\n" + "="*60)
print("RESULTS")
print("="*60)
print(f"RBQL:               {results['rbql_avg']:.1f} ± {results['rbql_std']:.1f} episodes")
print(f"Q-Learning (α=0.5): {results['standard_q_avg']:.1f} ± {results['standard_q_std']:.1f} episodes")
print(f"Q-Learning (α=1.0): {results['q_alpha1_avg']:.1f} ± {results['q_alpha1_std']:.1f} episodes")
print()

if results['q_alpha1_avg'] > 0:
    speedup_fair = results['q_alpha1_avg'] / results['rbql_avg']
    print(f"RBQL vs Q-Learning (α=1.0): {speedup_fair:.2f}x faster")
    print("  ^ Fair comparison (same effective learning rate)")
print()

if results['standard_q_avg'] > 0:
    speedup = results['standard_q_avg'] / results['rbql_avg']
    print(f"RBQL vs Q-Learning (α=0.5): {speedup:.2f}x faster")