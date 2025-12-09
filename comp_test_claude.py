"""
RBQL vs Standard Q-Learning - Randomized Pong Environment
"""

import numpy as np
import random
import matplotlib.pyplot as plt
from collections import defaultdict, deque
from scipy import stats
import time

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------
NUM_RUNS = 30
EPISODES_PER_RUN = 400
GAMMA = 0.95
ALPHA_QL = 0.1
EPSILON_START = 1.0
EPSILON_DECAY = 1.0 / 400.0  # Per step
WINDOW_SIZE = 20
SUCCESS_THRESHOLD = 0.9

NUM_STATES = 13 * 12 * 2 * 2 * 12
NUM_ACTIONS = 2  # 0: Left (-1), 1: Right (+1)

# ------------------------------------------------------------------
# ENVIRONMENT (Randomized initial conditions)
# ------------------------------------------------------------------
class PongEnvironment:
    def __init__(self):
        self.reset()
        
    def reset(self):
        # Randomized starting conditions
        self.x_racket = random.randint(0, 11)
        self.x_ball = random.randint(1, 10)
        self.y_ball = 1  # Start at top
        self.vx_ball = random.choice([-1, 1])
        self.vy_ball = 1  # Always moving down initially
        return self.get_state()

    def get_state(self):
        return int((((self.x_ball * 13 + self.y_ball) * 2 + (self.vx_ball + 1) // 2)
                    * 2 + (self.vy_ball + 1) // 2) * 12 + self.x_racket)

    def step(self, action_index):
        action = -1 if action_index == 0 else 1
        
        self.x_racket += action
        self.x_racket = max(0, min(11, self.x_racket))
        
        self.x_ball += self.vx_ball
        self.y_ball += self.vy_ball
        
        if self.x_ball > 10 or self.x_ball < 1:
            self.vx_ball *= -1
        if self.y_ball > 11 or self.y_ball < 1:
            self.vy_ball *= -1
            
        next_state = self.get_state()
        reward = 0
        done = False
        
        if self.y_ball == 12:
            done = True
            if self.x_ball >= self.x_racket and self.x_ball <= self.x_racket + 4:
                reward = 1
            else:
                reward = -1
                
        return next_state, reward, done

# ------------------------------------------------------------------
# RBQL AGENT
# ------------------------------------------------------------------
class PersistentModel:
    def __init__(self):
        self.explored_map = {}
        self.rewards = {}
    
    def add_transition(self, state, action_index, next_state, reward):
        if state not in self.explored_map:
            self.explored_map[state] = [None, None]
        self.explored_map[state][action_index] = next_state
        self.rewards[(state, action_index)] = reward
        
    def build_backward_graph(self):
        backward = defaultdict(list)
        for state, next_states in self.explored_map.items():
            for action_index, next_state in enumerate(next_states):
                if next_state is not None:
                    reward = self.rewards.get((state, action_index), 0)
                    backward[next_state].append((state, action_index, reward))
        return backward


class RBQLAgent:
    def __init__(self):
        self.q_values = np.random.rand(NUM_STATES, NUM_ACTIONS) / 1000.0
        self.model = PersistentModel()
        self.epsilon = EPSILON_START
        
    def get_action(self, state):
        if random.random() < self.epsilon:
            return random.choice([0, 1])
        return int(np.argmax(self.q_values[state]))
    
    def update(self, state, action_index, reward, next_state, done):
        self.model.add_transition(state, action_index, next_state, reward)
        if done:
            self._propagate_rewards(next_state)
            
    def _propagate_rewards(self, terminal_state):
        backward = self.model.build_backward_graph()
        
        visited = {terminal_state}
        queue = deque([terminal_state])
        updates = []
        
        while queue:
            current = queue.popleft()
            for prev_state, action_index, reward in backward[current]:
                updates.append((prev_state, action_index, current, reward))
                if prev_state not in visited:
                    visited.add(prev_state)
                    queue.append(prev_state)
        
        # Update in BFS order (closest to terminal first)
        for s, a, ns, r in updates:
            next_q = np.max(self.q_values[ns])
            self.q_values[s][a] = r + GAMMA * next_q
            
    def decay_epsilon(self):
        self.epsilon = max(0, self.epsilon - EPSILON_DECAY)

# ------------------------------------------------------------------
# Q-LEARNING AGENT
# ------------------------------------------------------------------
class QLearningAgent:
    def __init__(self):
        self.q_values = np.random.rand(NUM_STATES, NUM_ACTIONS) / 1000.0
        self.epsilon = EPSILON_START
        
    def get_action(self, state):
        if random.random() < self.epsilon:
            return random.choice([0, 1])
        return int(np.argmax(self.q_values[state]))
    
    def update(self, state, action_index, reward, next_state, done):
        if done:
            target = reward
        else:
            target = reward + GAMMA * np.max(self.q_values[next_state])
        
        self.q_values[state][action_index] += ALPHA_QL * (target - self.q_values[state][action_index])
        
    def decay_epsilon(self):
        self.epsilon = max(0, self.epsilon - EPSILON_DECAY)

# ------------------------------------------------------------------
# EXPERIMENT RUNNER
# ------------------------------------------------------------------
def run_experiment(agent_class, label):
    print(f"Running {label}...")
    all_rewards = []
    convergence_episodes = []
    
    for run in range(NUM_RUNS):
        env = PongEnvironment()
        agent = agent_class()
        rewards = []
        rolling_window = deque(maxlen=WINDOW_SIZE)
        converged_at = None
        
        for ep in range(EPISODES_PER_RUN):
            state = env.reset()
            done = False
            
            while not done:
                action = agent.get_action(state)
                next_state, reward, done = env.step(action)
                agent.update(state, action, reward, next_state, done)
                agent.decay_epsilon()
                state = next_state
            
            rewards.append(reward)
            rolling_window.append(1 if reward > 0 else 0)
            
            if converged_at is None and len(rolling_window) == WINDOW_SIZE:
                if sum(rolling_window) / WINDOW_SIZE >= SUCCESS_THRESHOLD:
                    converged_at = ep + 1
        
        all_rewards.append(rewards)
        convergence_episodes.append(converged_at if converged_at else EPISODES_PER_RUN)
        
        if (run + 1) % 10 == 0:
            print(f"  Run {run + 1}/{NUM_RUNS}, Conv: {convergence_episodes[-1]}")
    
    return np.array(all_rewards), np.array(convergence_episodes)

# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------
if __name__ == "__main__":
    start = time.time()
    
    rbql_rewards, rbql_conv = run_experiment(RBQLAgent, "RBQL")
    ql_rewards, ql_conv = run_experiment(QLearningAgent, "Q-Learning")
    
    # Results
    print("\n" + "="*50)
    print("RESULTS")
    print("="*50)
    print(f"RBQL:       {np.mean(rbql_conv):.1f} ± {np.std(rbql_conv):.1f} episodes")
    print(f"Q-Learning: {np.mean(ql_conv):.1f} ± {np.std(ql_conv):.1f} episodes")
    
    t_stat, p_val = stats.ttest_ind(rbql_conv, ql_conv)
    print(f"p-value:    {p_val:.4e}")
    print("Significant!" if p_val < 0.05 else "Not significant")
    print("="*50)
    
    # Plot
    def rolling_success(rewards_array, window=20):
        result = []
        for run in rewards_array:
            run_rolling = []
            for i in range(len(run)):
                start_idx = max(0, i - window + 1)
                w = run[start_idx:i+1]
                run_rolling.append(sum(1 for r in w if r > 0) / len(w))
            result.append(run_rolling)
        return np.array(result)
    
    rbql_roll = rolling_success(rbql_rewards)
    ql_roll = rolling_success(ql_rewards)
    
    episodes = range(1, EPISODES_PER_RUN + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Learning curve
    plt.subplot(1, 2, 1)
    plt.plot(episodes, rbql_roll.mean(axis=0), 'b-', label='RBQL', linewidth=2)
    plt.fill_between(episodes, 
                     rbql_roll.mean(axis=0) - rbql_roll.std(axis=0),
                     rbql_roll.mean(axis=0) + rbql_roll.std(axis=0), 
                     color='blue', alpha=0.2)
    plt.plot(episodes, ql_roll.mean(axis=0), 'r-', label='Q-Learning', linewidth=2)
    plt.fill_between(episodes,
                     ql_roll.mean(axis=0) - ql_roll.std(axis=0),
                     ql_roll.mean(axis=0) + ql_roll.std(axis=0),
                     color='red', alpha=0.2)
    plt.axhline(y=0.9, color='green', linestyle='--', alpha=0.7)
    plt.xlabel('Episode')
    plt.ylabel('Success Rate (rolling 20)')
    plt.title('Learning Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.05)
    
    # Convergence bar chart
    plt.subplot(1, 2, 2)
    means = [np.mean(rbql_conv), np.mean(ql_conv)]
    stds = [np.std(rbql_conv), np.std(ql_conv)]
    bars = plt.bar(['RBQL', 'Q-Learning'], means, yerr=stds, 
                   color=['blue', 'red'], capsize=10, edgecolor='black')
    plt.ylabel('Episodes to 90% Success')
    plt.title('Convergence Speed')
    for bar, m in zip(bars, means):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'{m:.0f}', ha='center')
    
    plt.tight_layout()
    plt.savefig('rbql_vs_qlearning.png', dpi=150)
    print(f"\nPlot saved. Time: {time.time() - start:.1f}s")
    plt.show()