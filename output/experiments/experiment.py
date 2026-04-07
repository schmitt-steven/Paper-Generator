import os
import sys
import time
import json
import numpy as np
from collections import deque

# Set headless environment variables for safety
os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['SDL_AUDIODRIVER'] = 'dummy'

# Set matplotlib backend to Agg for headless plotting
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# ENVIRONMENT DYNAMICS (DRY Principle)
# ============================================================

def get_state(x_ball, y_ball, vx_ball, vy_ball, x_racket):
    """Encodes the environment variables into a single discrete integer state."""
    return int((((x_ball * 13 + y_ball) * 2 + (vx_ball + 1) // 2) * 2 + (vy_ball + 1) // 2) * 12 + x_racket)

def step_env(x_racket, x_ball, y_ball, vx_ball, vy_ball, action):
    """
    Executes one step of the Pong-like environment.
    Returns: next_state, reward, done, new_env_vars
    """
    x_racket = np.clip(x_racket + action, 0, 11)
    x_ball += vx_ball
    y_ball += vy_ball

    if x_ball > 10 or x_ball < 1: vx_ball *= -1
    if y_ball > 11 or y_ball < 1: vy_ball *= -1

    reward = 0
    done = False

    if y_ball == 12:
        reward = 1 if (x_racket <= x_ball <= x_racket + 4) else -1
        done = True

    next_state = get_state(x_ball, y_ball, vx_ball, vy_ball, x_racket)
    return next_state, reward, done, (x_racket, x_ball, y_ball, vx_ball, vy_ball)

def reset_env():
    """Resets environment to initial state."""
    return 5, 1, 1, 1, 1

# ============================================================
# ALGORITHM IMPLEMENTATIONS
# ============================================================

def get_action(state, epsilon, Q):
    if np.random.rand() <= epsilon:
        return np.random.choice([-1, 1])
    return (np.argmax(Q[state]) * 2) - 1

def run_backward_prop(num_states=7488, num_actions=2, max_episodes=300, gamma=0.95):
    """
    Runs Recursive Backwards Q-Learning (RBQL).
    Builds a reverse transition map per episode, then propagates the discounted
    terminal reward backwards via BFS with direct Q-value assignment (alpha=1).
    """
    Q = np.random.rand(num_states, num_actions) / 1000.0
    Q_updated = np.zeros((num_states, num_actions), dtype=bool)
    epsilon = 1.0

    x_racket, x_ball, y_ball, vx_ball, vy_ball = reset_env()
    episode = 0
    episode_transitions = []
    run_rewards = []

    while episode < max_episodes:
        epsilon = max(0, epsilon - 1 / 400)

        state = get_state(x_ball, y_ball, vx_ball, vy_ball, x_racket)
        action = get_action(state, epsilon, Q)

        next_state, reward, done, (x_racket, x_ball, y_ball, vx_ball, vy_ball) = step_env(
            x_racket, x_ball, y_ball, vx_ball, vy_ball, action
        )

        action_idx = (action + 1) // 2
        episode_transitions.append((state, action_idx, next_state))

        if done:
            run_rewards.append(reward)
            episode += 1

            # Build reverse transition map for the episode
            reverse_map = {}
            for s, a, ns in episode_transitions:
                if ns not in reverse_map:
                    reverse_map[ns] = []
                reverse_map[ns].append((s, a))

            # Breadth-first backward propagation from terminal state
            q = deque()
            s_term, a_term, ns_term = episode_transitions[-1]
            q.append((s_term, a_term, reward))

            visited = set()
            while q:
                s, a, r = q.popleft()
                if not Q_updated[s, a]:
                    Q[s, a] = r
                    Q_updated[s, a] = True
                if s in reverse_map:
                    for s_prev, a_prev in reverse_map[s]:
                        if (s_prev, a_prev) not in visited:
                            visited.add((s_prev, a_prev))
                            q.append((s_prev, a_prev, r * gamma))

            episode_transitions = []

    return run_rewards

def update_q_per(reward, state, action, next_state, Q,
                 per_re, per_s, per_a, per_ns, per_prio,
                 tick, buf_fill, buffer_size, alpha, gamma, per_alpha, per_epsilon, batch_size):
    """
    Registers a new transition into the buffer and updates Q-values using prioritized sampling.
    """
    idx = tick % buffer_size

    max_prio = per_prio[:buf_fill].max() if buf_fill > 0 else 1.0
    per_re[idx] = reward
    per_s[idx]  = state
    per_a[idx]  = action
    per_ns[idx] = next_state
    per_prio[idx] = max_prio

    prios = per_prio[:buf_fill]
    probs = prios ** per_alpha
    prob_sum = probs.sum()
    if prob_sum > 0:
        probs /= prob_sum
    else:
        probs = np.ones_like(probs) / len(probs)

    sample_indices = np.random.choice(buf_fill, size=batch_size, replace=True, p=probs)

    for i in sample_indices:
        s  = int(per_s[i])
        a  = int(per_a[i])
        ns = int(per_ns[i])
        r  = per_re[i]

        td_error = r + gamma * np.max(Q[ns]) - Q[s, a]
        Q[s, a] += alpha * td_error

        per_prio[i] = abs(td_error) + per_epsilon

def run_prioritized_experience_replay(num_states=7488, num_actions=2, max_episodes=300, gamma=0.95,
                                      alpha_lr=0.1, batch_size=32, buffer_size=1000,
                                      per_alpha=0.6, per_epsilon=1e-4):
    """
    Runs Q-Learning with Prioritized Experience Replay as a model-free baseline comparison.
    """
    Q = np.random.rand(num_states, num_actions) / 1000.0

    per_re   = np.zeros(buffer_size)
    per_s    = np.zeros(buffer_size)
    per_a    = np.zeros(buffer_size)
    per_ns   = np.zeros(buffer_size)
    per_prio = np.ones(buffer_size) * per_epsilon

    epsilon = 1.0
    tick = 0

    x_racket, x_ball, y_ball, vx_ball, vy_ball = reset_env()
    episode = 0
    run_rewards = []

    while episode < max_episodes:
        epsilon = max(0, epsilon - 1 / 400)

        state = get_state(x_ball, y_ball, vx_ball, vy_ball, x_racket)
        action = get_action(state, epsilon, Q)

        next_state, reward, done, (x_racket, x_ball, y_ball, vx_ball, vy_ball) = step_env(
            x_racket, x_ball, y_ball, vx_ball, vy_ball, action
        )

        buf_fill = min(tick + 1, buffer_size)

        update_q_per(reward, state, (action + 1) // 2, next_state, Q,
                     per_re, per_s, per_a, per_ns, per_prio,
                     tick, buf_fill, buffer_size, alpha_lr, gamma, per_alpha, per_epsilon, batch_size)
        tick += 1

        if done:
            run_rewards.append(reward)
            episode += 1

    return run_rewards

# ============================================================
# EXPERIMENT EXECUTION & VISUALIZATION
# ============================================================

def run_experiment():
    os.makedirs('plots', exist_ok=True)

    # Hyperparameters
    NUM_SEEDS = 30
    MAX_EPISODES = 300
    GAMMA = 0.95
    CONV_WINDOW = 20
    CONV_THRESHOLD = 0.5

    # Storage for results
    all_rbql_rewards = []
    all_per_rewards = []
    rbql_times = []
    per_times = []

    print("=" * 80)
    print("EXPERIMENT: Recursive Backwards Q-Learning (RBQL) vs. Prioritized Experience Replay")
    print("=" * 80)

    # Run Experiments
    for seed in range(NUM_SEEDS):
        np.random.seed(seed)

        # Run RBQL
        start_time = time.time()
        rbql_rewards = run_backward_prop(max_episodes=MAX_EPISODES, gamma=GAMMA)
        rbql_time = time.time() - start_time

        # Reset seed for fair comparison
        np.random.seed(seed + 1000)

        # Run PER
        start_time = time.time()
        per_rewards = run_prioritized_experience_replay(max_episodes=MAX_EPISODES, gamma=GAMMA)
        per_time = time.time() - start_time

        all_rbql_rewards.append(rbql_rewards)
        all_per_rewards.append(per_rewards)
        rbql_times.append(rbql_time)
        per_times.append(per_time)

        print(f"  Seed {seed}: RBQL {rbql_time:.2f}s, PER {per_time:.2f}s")

    # ============================================================
    # CONVERGENCE DETECTION
    # ============================================================

    def convergence_episode(rewards):
        """First episode where moving avg stays above threshold permanently."""
        n = len(rewards)
        for i in range(CONV_WINDOW, n):
            window_mean = np.mean(rewards[i - CONV_WINDOW:i])
            if window_mean >= CONV_THRESHOLD:
                remaining = [np.mean(rewards[j - CONV_WINDOW:j]) for j in range(i, n)]
                if all(v >= CONV_THRESHOLD for v in remaining):
                    return i
        return MAX_EPISODES

    conv_rbql = [convergence_episode(r) for r in all_rbql_rewards]
    conv_per = [convergence_episode(r) for r in all_per_rewards]
    sum_rbql = [sum(r) for r in all_rbql_rewards]
    sum_per = [sum(r) for r in all_per_rewards]

    # ============================================================
    # METRICS TABLE
    # ============================================================

    print("\n[CONVERGENCE METRICS TABLE]")
    print("-" * 80)
    print(f"{'Metric':<30} {'RBQL (Mean +/- Std)':<25} {'PER (Mean +/- Std)':<25}")
    print("-" * 80)
    print(f"{'Convergence Episode':<30} "
          f"{np.mean(conv_rbql):>7.1f} +/- {np.std(conv_rbql):<10.1f} "
          f"{np.mean(conv_per):>7.1f} +/- {np.std(conv_per):<10.1f}")
    print(f"{'Total Reward Sum':<30} "
          f"{np.mean(sum_rbql):>7.1f} +/- {np.std(sum_rbql):<10.1f} "
          f"{np.mean(sum_per):>7.1f} +/- {np.std(sum_per):<10.1f}")
    print(f"{'Wall-Clock Time (s)':<30} "
          f"{np.mean(rbql_times):>7.2f} +/- {np.std(rbql_times):<10.2f} "
          f"{np.mean(per_times):>7.2f} +/- {np.std(per_times):<10.2f}")
    print("-" * 80)

    # ============================================================
    # SAVE RESULTS
    # ============================================================

    results = {
        "hyperparameters": {
            "gamma": GAMMA,
            "max_episodes": MAX_EPISODES,
            "seeds": NUM_SEEDS,
            "conv_window": CONV_WINDOW,
            "conv_threshold": CONV_THRESHOLD
        },
        "rbql": {
            "mean_time": float(np.mean(rbql_times)),
            "std_time": float(np.std(rbql_times)),
            "convergence_episodes": conv_rbql,
            "mean_convergence": float(np.mean(conv_rbql)),
            "total_rewards": sum_rbql,
            "mean_total_reward": float(np.mean(sum_rbql))
        },
        "per": {
            "mean_time": float(np.mean(per_times)),
            "std_time": float(np.std(per_times)),
            "convergence_episodes": conv_per,
            "mean_convergence": float(np.mean(conv_per)),
            "total_rewards": sum_per,
            "mean_total_reward": float(np.mean(sum_per))
        }
    }

    # ============================================================
    # SCALING EXPERIMENT (Wall-Clock Time vs. Episode Budget)
    # ============================================================

    SCALING_SEEDS = 5
    EPISODE_BUDGETS = [50, 100, 150, 200, 250, 300]

    rbql_scaling_times = []
    per_scaling_times = []

    print("\n[SCALING EXPERIMENT]")
    for budget in EPISODE_BUDGETS:
        rbql_t = []
        per_t = []
        for s in range(SCALING_SEEDS):
            np.random.seed(s + 2000)
            t0 = time.time()
            run_backward_prop(max_episodes=budget, gamma=GAMMA)
            rbql_t.append(time.time() - t0)

            np.random.seed(s + 3000)
            t0 = time.time()
            run_prioritized_experience_replay(max_episodes=budget, gamma=GAMMA)
            per_t.append(time.time() - t0)

        rbql_scaling_times.append(np.mean(rbql_t))
        per_scaling_times.append(np.mean(per_t))
        print(f"  Budget {budget}: RBQL {np.mean(rbql_t):.2f}s, PER {np.mean(per_t):.2f}s")

    results["scaling"] = {
        "episode_budgets": EPISODE_BUDGETS,
        "rbql_times": rbql_scaling_times,
        "per_times": per_scaling_times,
        "seeds_per_budget": SCALING_SEEDS
    }

    with open('results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # ============================================================
    # VISUALIZATION
    # ============================================================

    sns.set_theme(style="ticks", font_scale=1.1)
    pal = sns.color_palette("colorblind")
    C_RBQL, C_PER = pal[0], pal[1]

    # Compute per-seed statistics for learning curves
    min_len = min(min(len(r) for r in all_rbql_rewards),
                  min(len(r) for r in all_per_rewards))
    rbql_arr = np.array([r[:min_len] for r in all_rbql_rewards])
    per_arr = np.array([r[:min_len] for r in all_per_rewards])
    rbql_mean = rbql_arr.mean(axis=0)
    rbql_std = rbql_arr.std(axis=0)
    per_mean = per_arr.mean(axis=0)
    per_std = per_arr.std(axis=0)
    episodes_x = np.arange(1, min_len + 1)

    # Plot 1: Sample Complexity (Learning Curves with std bands)
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(episodes_x, rbql_mean, label='RBQL', color=C_RBQL, linewidth=2)
    ax.fill_between(episodes_x, rbql_mean - rbql_std, rbql_mean + rbql_std,
                    alpha=0.2, color=C_RBQL)

    ax.plot(episodes_x, per_mean, label='PER', color=C_PER,
            linestyle='--', linewidth=2)
    ax.fill_between(episodes_x, per_mean - per_std, per_mean + per_std,
                    alpha=0.15, color=C_PER)

    ax.axhline(0, color='gray', linestyle=':', linewidth=0.7)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Mean Reward')
    ax.set_title(f'Sample Complexity ({NUM_SEEDS} runs, \u00b11 std)')
    ax.legend(loc='lower right', frameon=True)
    sns.despine()

    plt.tight_layout()
    plt.savefig('plots/sample_complexity.pdf', format='pdf')
    plt.close()

    # Plot 2: Boxplots (Total Reward Sum + Convergence Episode)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    for ax_b, data, title, ylabel in [
        (ax1, [sum_rbql, sum_per],
         f'Total Reward ({MAX_EPISODES} episodes)', 'Cumulative Reward'),
        (ax2, [conv_rbql, conv_per],
         f'Convergence Episode\n(window={CONV_WINDOW}, thr={CONV_THRESHOLD})', 'Episode'),
    ]:
        bp = ax_b.boxplot(
            data, labels=['RBQL', 'PER'], patch_artist=True,
            medianprops=dict(color='black', linewidth=1.5), widths=0.4,
        )
        bp['boxes'][0].set(facecolor=C_RBQL, alpha=0.7)
        bp['boxes'][1].set(facecolor=C_PER, alpha=0.7, hatch='...')
        ax_b.set_title(title)
        ax_b.set_ylabel(ylabel)
        ax_b.grid(axis='y', alpha=0.3, linestyle=':')

    ax2.set_ylim(0, MAX_EPISODES + 10)
    sns.despine()
    plt.tight_layout()
    plt.savefig('plots/boxplots.pdf', format='pdf')
    plt.close()

    # Plot 3: Wall-Clock Time (bar)
    fig, ax = plt.subplots(figsize=(5, 5))

    methods = ['RBQL', 'PER']
    times = [np.mean(rbql_times), np.mean(per_times)]
    bars = ax.bar(methods, times, color=[C_RBQL, C_PER],
                  edgecolor='black', linewidth=0.6, width=0.5)
    bars[1].set_hatch('...')

    ax.set_ylabel('Time (seconds)')
    ax.set_title('Mean Wall-Clock Time')
    ax.set_ylim(0, max(times) * 1.15 if times else 1.0)

    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                f'{t:.2f}s', ha='center', va='bottom', fontsize=9)

    sns.despine()
    plt.tight_layout()
    plt.savefig('plots/wall_clock_time.pdf', format='pdf')
    plt.close()

    # Plot 4: Wall-Clock Time Scaling (line graph)
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(EPISODE_BUDGETS, rbql_scaling_times, 'D-', label='RBQL',
            color=C_RBQL, linewidth=2, markersize=6)
    ax.plot(EPISODE_BUDGETS, per_scaling_times, '^--', label='PER',
            color=C_PER, linewidth=2, markersize=6)

    ax.set_xlabel('Episode Budget')
    ax.set_ylabel('Wall-Clock Time (seconds)')
    ax.set_title(f'Time Scaling ({SCALING_SEEDS} seeds per point)')
    ax.legend(frameon=True)
    sns.despine()

    plt.tight_layout()
    plt.savefig('plots/wall_clock_scaling.pdf', format='pdf')
    plt.close()

    print("\n[EXPERIMENT COMPLETE]")
    print(f"Results saved to 'results.json'")
    print(f"Plots saved to 'plots/' directory")

if __name__ == "__main__":
    run_experiment()
