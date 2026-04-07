import numpy as np
from collections import deque

def get_state(x_ball, y_ball, vx_ball, vy_ball, x_racket):
    """Encodes the environment variables into a single discrete integer state."""
    return int((((x_ball * 13 + y_ball) * 2 + (vx_ball + 1) // 2) * 2 + (vy_ball + 1) // 2) * 12 + x_racket)

# ============================================================
# ALGORITHM 1: Recursive Backwards Q-Learning (RBQL)
# ============================================================

def get_action_bp(state, epsilon, Q):
    if np.random.rand() <= epsilon:
        return np.random.choice([-1, 1])
    return (np.argmax(Q[state]) * 2) - 1

def run_backward_prop(num_states=7488, num_actions=2, max_episodes=300, gamma=0.95):
    """
    Runs Recursive Backwards Q-Learning. 
    Constructs an episodic trajectory graph and traverses it backwards upon terminal rewards.
    """
    Q = np.random.rand(num_states, num_actions) / 1000.0
    Q_updated = np.zeros((num_states, num_actions), dtype=bool)
    epsilon = 1.0

    x_racket, x_ball, y_ball, vx_ball, vy_ball = 5, 1, 1, 1, 1
    episode = 0
    episode_transitions = []
    run_rewards = []

    while episode < max_episodes:
        epsilon = max(0, epsilon - 1 / 400)

        state = get_state(x_ball, y_ball, vx_ball, vy_ball, x_racket)
        action = get_action_bp(state, epsilon, Q)

        # Environment dynamics
        x_racket = np.clip(x_racket + action, 0, 11)
        x_ball += vx_ball
        y_ball += vy_ball
        if x_ball > 10 or x_ball < 1: vx_ball *= -1
        if y_ball > 11 or y_ball < 1: vy_ball *= -1

        reward = 0
        if y_ball == 12:
            reward = 1 if (x_racket <= x_ball <= x_racket + 4) else -1
            episode += 1

        next_state = get_state(x_ball, y_ball, vx_ball, vy_ball, x_racket)
        action_idx = (action + 1) // 2
        episode_transitions.append((state, action_idx, next_state))

        if reward != 0:
            run_rewards.append(reward)

            # Build reverse transition map for the episode
            reverse_map = {}
            for s, a, ns in episode_transitions:
                if ns not in reverse_map:
                    reverse_map[ns] = []
                reverse_map[ns].append((s, a))

            # Breadth-first backward propagation
            q = deque()
            s_term, a_term, ns_term = episode_transitions[-1]
            q.append((s_term, a_term, reward))

            visited = set()
            while q:
                s, a, r = q.popleft()
                
                # Full replacement update exactly once
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

# ============================================================
# ALGORITHM 2: Prioritized Experience Replay (Baseline)
# ============================================================

def get_action_per(state, epsilon, Q):
    if np.random.rand() <= epsilon:
        return np.random.choice([-1, 1])
    return (np.argmax(Q[state]) * 2) - 1

def update_q_per(reward, state, action, next_state, Q,
                 per_re, per_s, per_a, per_ns, per_prio,
                 tick, buf_fill, buffer_size, alpha, gamma, per_alpha, per_epsilon, batch_size):
    """
    Registers a new transition into the buffer and updates Q-values using prioritized sampling.
    """
    idx = tick % buffer_size

    # New transition gets the maximum historical priority to guarantee it gets sampled at least once
    max_prio = per_prio[:buf_fill].max() if buf_fill > 0 else 1.0
    per_re[idx] = reward
    per_s[idx]  = state
    per_a[idx]  = action
    per_ns[idx] = next_state
    per_prio[idx] = max_prio

    # Sample transitions proportionally to priority^per_alpha
    prios = per_prio[:buf_fill]
    probs = prios ** per_alpha
    probs /= probs.sum()

    sample_indices = np.random.choice(buf_fill, size=batch_size, replace=True, p=probs)

    for i in sample_indices:
        s  = int(per_s[i])
        a  = int(per_a[i])
        ns = int(per_ns[i])
        r  = per_re[i]

        td_error = r + gamma * np.max(Q[ns]) - Q[s, a]
        Q[s, a] += alpha * td_error

        # Update priority for the sampled transition based on the TD-error
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

    x_racket, x_ball, y_ball, vx_ball, vy_ball = 5, 1, 1, 1, 1
    episode = 0
    run_rewards = []

    while episode < max_episodes:
        epsilon = max(0, epsilon - 1 / 400)

        state = get_state(x_ball, y_ball, vx_ball, vy_ball, x_racket)
        action = get_action_per(state, epsilon, Q)

        # Environment dynamics
        x_racket = np.clip(x_racket + action, 0, 11)
        x_ball += vx_ball
        y_ball += vy_ball
        if x_ball > 10 or x_ball < 1: vx_ball *= -1
        if y_ball > 11 or y_ball < 1: vy_ball *= -1

        reward = 0
        if y_ball == 12:
            reward = 1 if (x_racket <= x_ball <= x_racket + 4) else -1
            episode += 1

        next_state = get_state(x_ball, y_ball, vx_ball, vy_ball, x_racket)
        buf_fill = min(tick + 1, buffer_size)

        update_q_per(reward, state, (action + 1) // 2, next_state, Q,
                     per_re, per_s, per_a, per_ns, per_prio,
                     tick, buf_fill, buffer_size, alpha_lr, gamma, per_alpha, per_epsilon, batch_size)
        tick += 1

        if reward != 0:
            run_rewards.append(reward)

    return run_rewards
