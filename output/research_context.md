# Research Context

## 1. Keywords
- **Primary Domain:** Reinforcement Learning and Decision Making
- **Specific Task:** Sample-Efficient Policy Optimization in Deterministic Episodic Environments with Sparse Rewards
- **Methodological Class:** Model-Based Recursive Backwards Q-Learning via Breadth-First Search

## 2. Research Direction & Scope
Standard model-free Q-learning often exhibits prohibitive sample complexity in deterministic episodic settings, as sparse terminal rewards require repeated stochastic visits to propagate value signals backward through the state space. This work investigates a proposed mechanism termed Recursive Backwards Q-Learning (RBQL), which constructs a persistent state-transition trajectory graph during exploration and executes a breadth-first search from terminal states upon episode termination. The method leverages exact Bellman optimality updates with full replacement ($\alpha=1$) to propagate rewards through the reverse transition map, theoretically allowing value signals to reach initial states in a single pass per episode. Preliminary analysis suggests this approach aims to eliminate the iterative online update cycles inherent in standard Q-learning and Prioritized Experience Replay (PER), potentially reducing sample complexity by orders of magnitude while maintaining convergence guarantees for discrete deterministic Markov Decision Processes.

## 3. Problem Definition
- **The Bottleneck:** The fundamental limitation is the delayed propagation of sparse terminal rewards through long trajectories, necessitating a high number of repeated environment interactions to achieve value convergence in standard model-free approaches.
- **The Constraint:** This inefficiency is strictly exacerbated within perfectly deterministic environments where stochasticity cannot be relied upon for exploration, and where continuous state spaces or neural function approximators are excluded from the solution space.

## 4. Technical Approach
- **Architecture:** A tabular Q-learning framework augmented with an episode-specific reverse transition map (DAG) that facilitates a breadth-first traversal of the trajectory graph starting exclusively from terminal states.
- **Key differentiator:** Unlike standard online updates or PER's priority sampling based on TD-error magnitude, this implementation employs synchronous Bellman updates with full replacement ($\alpha=1$) to ensure each state-action pair is updated exactly once per episode, bypassing the need for iterative value iteration loops.



# Open Questions for Literature Search

### Related Work & Prior Art
1. How do existing model-based dynamic programming methods (e.g., Real-Time Dynamic Programming, RTDP) and backward induction techniques specifically handle sparse terminal rewards in deterministic episodic MDPs compared to the proposed RBQL's single-pass BFS propagation?
2. What are the theoretical convergence bounds for tabular Q-learning with full replacement ($\alpha=1$) versus standard TD-learning ($0 < \alpha < 1$) when applied to deterministic environments, and how do these bounds explain the sample complexity differences in sparse reward settings?
3. In what ways does Prioritized Experience Replay (PER) fundamentally differ from RBQL regarding the mechanism of value propagation—specifically comparing PER's stochastic TD-error sampling against RBQL's deterministic reverse-graph traversal—and which approach has been proven superior for long-horizon sparse reward problems?

### Differentiation & Positioning
4. How does the proposed RBQL method technically distinguish itself from "Value Iteration on a DAG" or "Backward Induction" algorithms that explicitly construct state-transition graphs, particularly regarding the timing of graph construction (online vs. offline) and the handling of non-terminal states?
5. What specific advantages does RBQL's synchronous Bellman update with full replacement offer over asynchronous updates in terms of preventing value oscillation or divergence in deterministic environments where stochastic exploration is absent?

### Key Concepts & Background
6. What are the precise mathematical conditions required for a discrete MDP to guarantee that a single backward pass from terminal states (via BFS) achieves global optimality without requiring iterative re-evaluation, and how do these conditions relate to the concept of "deterministic reachability"?
7. How is the state-space complexity of maintaining a persistent reverse transition map in RBQL formally analyzed against the memory overhead of PER's replay buffer, particularly as the episode length increases?



# Important Code Snippets

## File: RBQL_algorithms.py

**Summary:** This file implements two tabular Q-learning algorithms for a discrete Pong-like environment: Recursive Backwards Q-Learning (RBQL) and Prioritized Experience Replay (PER). RBQL propagates rewards backwards through the episode trajectory using a reverse transition map, whereas PER updates Q-values via prioritized sampling from a replay buffer based on TD-error.

**Keywords:** Q-Learning, Recursive Backwards Q-Learning, Prioritized Experience Replay, Tabular Reinforcement Learning, TD-error, Breadth-First Search, Discrete State Space

**Method:** Implements two distinct Q-learning variants: RBQL constructs a reverse transition graph per episode to propagate terminal rewards backwards via BFS, while PER uses a fixed-size buffer with priority sampling based on TD-error magnitude.

**Contribution:** Implements and compares Recursive Backwards Q-Learning (RBQL) against Prioritized Experience Replay (PER) in a discrete Pong-like environment to evaluate sample efficiency and convergence speed.

**Code Snippets (2):**

### RBQL Backward Propagation via BFS
```python
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
```

### PER Priority Sampling and TD-Error Update
```python
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
```

---
