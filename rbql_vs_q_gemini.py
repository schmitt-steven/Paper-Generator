"""
Comparison Test: RBQL vs Standard Q-Learning in Pong Environment
"""

import numpy as np
import random
import matplotlib.pyplot as plt
from collections import defaultdict, deque
import time
import argparse
import sys
import pygame as pyg

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------
NUM_RUNS = 30
EPISODES_PER_RUN = 400
GAMMA = 0.95
ALPHA_QL = 0.1
EPSILON_START = 1.0
EPSILON_DECAY = 1.0 / 400.0 # Decay per step
WINDOW_SIZE = 20
SUCCESS_THRESHOLD = 0.9

NUM_STATES = 13 * 13 * 2 * 2 * 12
NUM_ACTIONS = 2  # 0: Left (-1), 1: Right (+1)

# ------------------------------------------------------------------
# ENVIRONMENT
# ------------------------------------------------------------------
class PongEnvironment:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.x_racket = 5
        self.x_ball = random.randint(1, 11) # Random start x
        self.y_ball = 1
        self.vx_ball = random.choice([-1, 1])
        self.vy_ball = 1 
        self.score = 0
        return self.get_state()

    def get_state(self):
        """Deterministic mapping to discrete state index."""
        return int((((self.x_ball * 13 + self.y_ball) * 2 + (self.vx_ball + 1) // 2)
                    * 2 + (self.vy_ball + 1) // 2) * 12 + self.x_racket)

    def step(self, action_index):
        """
        Action index: 0 -> -1 (Left), 1 -> +1 (Right)
        Returns: next_state, reward, done
        """
        action = -1 if action_index == 0 else 1
        
        # Move racket
        self.x_racket += action
        self.x_racket = max(0, min(11, self.x_racket))
        
        # Move ball
        self.x_ball += self.vx_ball
        self.y_ball += self.vy_ball
        
        # Wall collisions
        if self.x_ball > 10 or self.x_ball < 1:
            self.vx_ball *= -1
        if self.y_ball > 11 or self.y_ball < 1:
            self.vy_ball *= -1
            
        next_state = self.get_state()
        reward = 0
        done = False
        
        # Terminal condition
        if self.y_ball == 12:
            done = True
            # Check collision
            if self.x_ball >= self.x_racket and self.x_ball <= self.x_racket + 4:
                reward = 1
            else:
                reward = -1
                
        return next_state, reward, done

    def render(self, agent_label=""):
        # Pygame initialization handled outside or lazily? 
        # Better to init outside if possible, but for simplicity let's rely on global init if visual.
        
        # We need a screen surface - let's assume global 'screen' or create one lazily
        global screen, pygame_font
        
        screen.fill((0, 0, 0))
        
        # Score
        t = pygame_font.render(f"Score:{self.score}", True, (255, 255, 255))
        screen.blit(t, t.get_rect(centerx=screen.get_rect().centerx, top=10))
        
        # Agent Label
        if agent_label:
            l = pygame_font.render(f"Agent: {agent_label}", True, (255, 255, 0))
            screen.blit(l, l.get_rect(centerx=screen.get_rect().centerx, top=30))
        
        # draw objects
        pyg.draw.rect(screen, (0, 128, 255), pyg.Rect(self.x_racket*20, 250, 80, 10))
        pyg.draw.rect(screen, (255, 100, 0),   pyg.Rect(self.x_ball*20, self.y_ball*20, 20, 20))
        
        pyg.display.flip()

# ------------------------------------------------------------------
# RBQL AGENT
# ------------------------------------------------------------------
class PersistentModel:
    def __init__(self):
        # Forward model: state -> [next_state_action_0, next_state_action_1]
        self.explored_map = {}
        # Rewards: (state, action) -> reward
        self.rewards = {}
    
    def add_transition(self, state, action_index, next_state, reward):
        if state not in self.explored_map:
            self.explored_map[state] = [None, None]
        self.explored_map[state][action_index] = next_state
        self.rewards[(state, action_index)] = reward
        
    def get_reward(self, state, action_index):
        return self.rewards.get((state, action_index), 0.0)

    def build_backward_graph(self):
        backward = defaultdict(list)
        for state, next_states in self.explored_map.items():
            for action_index, next_state in enumerate(next_states):
                if next_state is not None:
                    reward = self.get_reward(state, action_index)
                    backward[next_state].append((state, action_index, reward))
        return backward

class RBQLAgent:
    def __init__(self):
        self.q_values = np.random.rand(NUM_STATES, NUM_ACTIONS) / 1000.0
        self.model = PersistentModel()
        self.epsilon = EPSILON_START
        
    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.choice([0, 1])
        return int(np.argmax(self.q_values[state]))
    
    def update(self, state, action_index, reward, next_state, done):
        # Add to persistent model
        self.model.add_transition(state, action_index, next_state, reward)
        
        if done:
            self._propagate_rewards(next_state)
            
    def _propagate_rewards(self, terminal_state):
        backward = self.model.build_backward_graph()
        
        # BFS to order states by distance from terminal
        # This ensures we update states closer to terminal first (reverse of trajectory)
        # But wait, logic in RBQL file says: "BFS discovery order = closest to terminal first. 
        # This ensures Q(s') is already updated before computing Q(s)."
        
        visited_states = set([terminal_state])
        queue = deque([terminal_state])
        updates = [] 
        
        while queue:
            current_state = queue.popleft()
            
            for prev_state, action_index, reward in backward[current_state]:
                # We record the update: (prev_state, action, current_state(=next), reward)
                # We process them in order of discovery
                updates.append((prev_state, action_index, current_state, reward))
                
                if prev_state not in visited_states:
                    visited_states.add(prev_state)
                    queue.append(prev_state)
        
        # Apply updates in BFS order
        for s, a, ns, r in updates:
            next_q = np.max(self.q_values[ns])
            # RBQL update: alpha=1
            self.q_values[s][a] = r + GAMMA * next_q
            
    def decay_epsilon(self):
        self.epsilon -= EPSILON_DECAY
        if self.epsilon < 0:
            self.epsilon = 0

# ------------------------------------------------------------------
# Q-LEARNING AGENT
# ------------------------------------------------------------------
class QLearningAgent:
    def __init__(self):
        self.q_values = np.random.rand(NUM_STATES, NUM_ACTIONS) / 1000.0
        self.epsilon = EPSILON_START
        
    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.choice([0, 1])
        return int(np.argmax(self.q_values[state]))
    
    def update(self, state, action_index, reward, next_state, done):
        current_q = self.q_values[state][action_index]
        
        if done:
            target = reward # Terminal state value is 0
        else:
            best_next_q = np.max(self.q_values[next_state])
            target = reward + GAMMA * best_next_q
            
        # Standard TD update
        self.q_values[state][action_index] = current_q + ALPHA_QL * (target - current_q)
        
    def decay_epsilon(self):
        self.epsilon -= EPSILON_DECAY
        if self.epsilon < 0:
            self.epsilon = 0

# ------------------------------------------------------------------
# EXPERIMENT RUNNER
# ------------------------------------------------------------------
def run_experiment_series(agent_class, label):
    print(f"Starting {label} runs...")
    all_rewards = []
    convergence_episodes = []
    
    for r in range(NUM_RUNS):
        env = PongEnvironment()
        agent = agent_class()
        run_rewards = []
        
        converged_at = None
        rolling_window = deque(maxlen=WINDOW_SIZE)
        
        for ep in range(EPISODES_PER_RUN):
            state = env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action = agent.get_action(state)
                next_state, reward, done = env.step(action)
                agent.update(state, action, reward, next_state, done)
                agent.decay_epsilon()
                state = next_state
                episode_reward += reward
            run_rewards.append(episode_reward)
            
            # Check convergence
            # Success is +1 (reward=1). Fail is -1.
            # Convert reward to success (1 if 1, 0 if -1) for simpler tracking?
            # Or just check if average reward >= 0.8 (since range is -1 to 1)
            # Threshold 0.9 success rate means 90% are +1. 
            # Avg reward = 0.9*1 + 0.1*(-1) = 0.8
            
            is_success = 1 if episode_reward > 0 else 0
            rolling_window.append(is_success)
            
            if converged_at is None and len(rolling_window) == WINDOW_SIZE:
                success_rate = sum(rolling_window) / WINDOW_SIZE
                if success_rate >= SUCCESS_THRESHOLD:
                    converged_at = ep + 1
                    
        if converged_at is None:
            converged_at = EPISODES_PER_RUN # Did not converge
            
        all_rewards.append(run_rewards)
        convergence_episodes.append(converged_at)
        
        if (r+1) % 5 == 0:
            print(f"  Run {r+1}/{NUM_RUNS} done. Conv: {converged_at}")
            
    return np.array(all_rewards), np.array(convergence_episodes)

def run_visual_demo(agent_class, label):
    print(f"\n--- STARTING VISUAL DEMO: {label} ---")
    
    # Init Pygame
    global screen, pygame_font
    pyg.init()
    screen = pyg.display.set_mode((240, 260))
    pygame_font = pyg.font.SysFont("arial", 15)
    clock = pyg.time.Clock()
    
    env = PongEnvironment()
    agent = agent_class()
    
    # Run for 400 episodes or until user closes
    for ep in range(1, 401):
        state = env.reset()
        done = False
        step_count = 0
        
        while not done:
            # Handle quit
            for event in pyg.event.get():
                if event.type == pyg.QUIT:
                    pyg.quit()
                    return
            
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state, done)
            agent.decay_epsilon()
            
            state = next_state
            
            # Render
            env.render(agent_label=label)
            clock.tick(60) # 60 FPS
            step_count += 1
            
        print(f"Episode {ep} finished. Score: {env.score}. Steps: {step_count}. Epsilon: {agent.epsilon:.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--visual", action="store_true", help="Run visual demonstration instead of full statistical test")
    args = parser.parse_args()

    if args.visual:
        # Run visual demo
        run_visual_demo(RBQLAgent, "RBQL")
        run_visual_demo(QLearningAgent, "Q-Learning")
        sys.exit(0)

    start_time = time.time()
    
    # Run RBQL
    rbql_rewards, rbql_conv = run_experiment_series(RBQLAgent, "RBQL")
    
    # Run Q-Learning
    ql_rewards, ql_conv = run_experiment_series(QLearningAgent, "Q-Learning")
    
    # Statistics
    rbql_mean_conv = np.mean(rbql_conv)
    rbql_std_conv = np.std(rbql_conv)
    ql_mean_conv = np.mean(ql_conv)
    ql_std_conv = np.std(ql_conv)
    
    print("\n--- RESULTS ---")
    print(f"RBQL Convergence: {rbql_mean_conv:.2f} +/- {rbql_std_conv:.2f} episodes")
    print(f"QL   Convergence: {ql_mean_conv:.2f} +/- {ql_std_conv:.2f} episodes")
    
    # Simple T-test
    from scipy import stats
    try:
        t_stat, p_val = stats.ttest_ind(rbql_conv, ql_conv)
        print(f"T-test: t={t_stat:.4f}, p={p_val:.4e}")
        if p_val < 0.05:
            print("Difference is statistically significant.")
        else:
            print("Difference is NOT statistically significant.")
    except ImportError:
        print("Scipy not found, skipping t-test.")
    
    # Plotting
    try:
        # Calculate success rate curve (average over runs)
        # Map rewards (-1, 1) to success (0, 1)
        rbql_success = (rbql_rewards + 1) / 2
        ql_success = (ql_rewards + 1) / 2
        
        rbql_avg = np.mean(rbql_success, axis=0)
        rbql_std = np.std(rbql_success, axis=0)
        ql_avg = np.mean(ql_success, axis=0)
        ql_std = np.std(ql_success, axis=0)
        
        # Rolling average for smoother plots
        def rolling_avg(a, n=20):
            ret = np.cumsum(a, dtype=float)
            ret[n:] = ret[n:] - ret[:-n]
            return ret[n - 1:] / n
            
        episodes = np.arange(1, EPISODES_PER_RUN + 1)
        plot_len = len(rolling_avg(rbql_avg))
        plot_episodes = episodes[WINDOW_SIZE-1:]
        
        plt.figure(figsize=(10, 6))
        
        # RBQL
        r_mean = rolling_avg(rbql_avg)
        r_std = rolling_avg(rbql_std) # Approximation
        plt.plot(plot_episodes, r_mean, label=f'RBQL (Mean Conv: {rbql_mean_conv:.1f})', color='blue')
        plt.fill_between(plot_episodes, r_mean - r_std*0.2, r_mean + r_std*0.2, color='blue', alpha=0.1)
        
        # QL
        q_mean = rolling_avg(ql_avg)
        q_std = rolling_avg(ql_std)
        plt.plot(plot_episodes, q_mean, label=f'Q-Learning (Mean Conv: {ql_mean_conv:.1f})', color='red')
        plt.fill_between(plot_episodes, q_mean - q_std*0.2, q_mean + q_std*0.2, color='red', alpha=0.1)
        
        plt.axhline(y=0.9, color='green', linestyle='--', label='Success Threshold (0.9)')
        plt.xlabel('Episode')
        plt.ylabel('Success Rate (Rolling avg 20)')
        plt.title('RBQL vs Standard Q-Learning: Pong Environment')
        plt.legend()
        plt.grid(True)
        
        plt.savefig('comparison_plot.png')
        print("Plot saved to comparison_plot.png")

        # 2. Convergence Speed Bar Chart
        plt.figure(figsize=(8, 6))
        labels = ['RBQL', 'Q-Learning']
        means = [rbql_mean_conv, ql_mean_conv]
        stds = [rbql_std_conv, ql_std_conv]
        
        plt.bar(labels, means, yerr=stds, capsize=10, color=['blue', 'red'], alpha=0.7)
        plt.ylabel('Episodes to Convergence')
        plt.title('Convergence Speed (Mean +/- Std)')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add values on top
        for i, v in enumerate(means):
            plt.text(i, v + 5, f"{v:.1f}", ha='center', fontweight='bold')
            
        plt.savefig('convergence_plot.png')
        print("Plot saved to convergence_plot.png")
        
    except Exception as e:
        print(f"Plotting failed: {e}")
        
    print(f"Total time: {time.time() - start_time:.2f}s")
