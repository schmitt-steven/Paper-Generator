# Paper Concept

## 1. Taxonomic Classification  
- **Primary Domain:** Reinforcement Learning  
- **Specific Task:** Deterministic Episodic Markov Decision Processes  
- **Methodological Class:** Model-Based Q-Learning with Backward Induction via BFS  

## 2. Abstract & Core Contribution  
Standard Q-learning in deterministic episodic environments suffers from slow convergence due to incremental, sample-based value updates requiring repeated state-action visits. Recursive Backwards Q-Learning (RBQL) overcomes this by maintaining a persistent transition model and performing a single backward induction pass via breadth-first search from terminal states upon episode completion. The algorithm constructs a reverse state-transition graph, then applies the Bellman optimality equation with α=1 to update all known Q-values in topological reverse order, enabling exact value propagation without iterative sampling. This mechanism eliminates the need for repeated environmental interactions to propagate rewards, resulting in significantly faster convergence to optimal policies under determinism.  

## 3. Problem Definition  
- **The Bottleneck:** Standard Q-learning updates Q-values incrementally via single-step temporal difference learning, requiring multiple visits to a state-action pair to propagate terminal rewards backward through the state space—a process exponentially inefficient in deterministic environments where transitions are reproducible and fully observable.  
- **The Constraint:** The method is constrained to deterministic, episodic environments with discrete state and action spaces, where transitions and rewards are fully observable upon execution.  

## 4. Technical Approach  
- **Architecture:** A dual-component framework combining an episodic transition model (storing (s, a) → (s', r)) with a backward propagation engine that performs breadth-first search over the inverse transition graph.  
- **Key differentiator:** Replaces incremental Q-learning updates with a global, single-pass Bellman update applied in reverse topological order from terminal states using α=1 (full replacement), leveraging the deterministic structure to compute exact optimal Q-values in one sweep per episode—unlike Dyna-Q or Monte Carlo methods, which rely on iterative sampling or averaging.  

---  
*Note: The implementation strictly enforces backward induction via BFS over the explored transition graph, with Q-value updates derived directly from the Bellman optimality equation applied in reverse chronological order of state discovery. No bootstrapping or averaging is used—updates are deterministic and exhaustive within the explored subspace.*

# Important Code Snippets

## File: recursive_backwards_q_learning.py

**Summary:** RBQL is a variant of Q-learning that maintains a persistent model of all observed transitions and performs backward value propagation via BFS from terminal states, updating Q-values for all visited states in one episode. This eliminates the need for repeated sampling and enables exact convergence on deterministic problems by leveraging full state-space knowledge.

**Keywords:** Q-learning, backward induction, BFS, persistent model, deterministic MDP

**Method:** Uses BFS over a persistent transition graph to update all known state-action values in reverse order from terminal states using the Bellman equation with α=1

**Contribution:** Core learning algorithm enabling global value propagation in deterministic environments

**Code Snippets (2):**

### Build Backward Graph
```python
def build_backward_graph(self):
        backward = defaultdict(list)
        for state, next_states in self.explored_map.items():
            for action_index, next_state in enumerate(next_states):
                if next_state is not None:
                    reward = self.get_reward(state, action_index)
                    backward[next_state].append((state, action_index, reward))
        return backward
```

### Backward Propagation with Bellman Update
```python
def propagate_reward_rbql(terminal_state):
    global q_values, gamma

    backward = model.build_backward_graph()

    visited_states = set([terminal_state])
    queue = deque([terminal_state])
    state_order = []

    while queue:
        current_state = queue.popleft()

        for state, action_index, reward in backward[current_state]:
            state_order.append((state, action_index, current_state, reward))

            if state not in visited_states:
                visited_states.add(state)
                queue.append(state)

    for state, action_index, next_state, reward in state_order:
        next_q = np.max(q_values[next_state])
        q_values[state][action_index] = reward + gamma * next_q
```

---

# Open Questions

1. **What existing model-based Q-learning variants (e.g., Dyna-Q, MBQL, R-MAX) in deterministic episodic MDPs perform global value updates, and how do their transition model usage, update scheduling, or backup strategies differ from RBQL’s single-pass backward BFS with α=1?**  
*(Targets: Prior art in model-based Q-learning; establishes novelty by contrasting update mechanics)*

2. **How do Monte Carlo methods (e.g., episodic MC control) and dynamic programming approaches (e.g., value iteration) in deterministic environments handle backward reward propagation, and why do they still require multiple episodes or full state-space knowledge—unlike RBQL’s on-the-fly BFS propagation from terminal states?**  
*(Targets: Baseline comparison against non-incremental methods; clarifies RBQL’s unique blend of model-based efficiency and online applicability)*

3. **What theoretical guarantees exist for convergence in deterministic MDPs under Bellman updates with α=1 applied in non-chronological or partial state-space orders, and how does RBQL’s BFS-induced topological ordering ensure optimality without full environment mapping?**  
*(Targets: Foundational theory; justifies correctness of backward propagation under partial exploration)*

4. **In deterministic episodic MDPs, what prior work has used backward induction via BFS over an inferred transition graph to update Q-values in reverse order, and what are the known limitations of such approaches in terms of memory, scalability, or partial observability?**  
*(Targets: Direct prior art search; isolates RBQL’s specific innovation in using BFS for Q-update sequencing)*

5. **How does RBQL’s use of α=1 (full replacement) and deterministic backward induction differ technically from TD(0), Dyna-Q, or Q-learning with experience replay in terms of bias-variance tradeoff and sample efficiency under determinism?**  
*(Targets: Technical differentiation; quantifies advantage over incremental methods)*

6. **What is the standard taxonomy of model-based RL algorithms in deterministic settings, and where does RBQL fit within categories such as “planning after learning,” “real-time dynamic programming,” or “episodic value propagation”?**  
*(Targets: Positioning within field taxonomy; clarifies conceptual novelty)*

7. **What are the canonical definitions of “deterministic episodic MDP,” “backward induction,” and “topological ordering” in reinforcement learning literature, and how do they constrain or enable the design of RBQL’s update mechanism?**  
*(Targets: Foundational terminology; ensures precise framing of assumptions and contributions)*

8. **Has any prior work combined persistent transition modeling with BFS-based backward propagation in Q-learning for deterministic environments, and if so, what were the reasons it was not adopted or failed to outperform incremental methods?**  
*(Targets: Gap analysis; identifies why RBQL’s approach is novel or overlooked)*

9. **How do state-space coverage and transition graph completeness affect the convergence speed of RBQL compared to standard Q-learning, and what are the theoretical bounds on the number of episodes required for full optimality under deterministic dynamics?**  
*(Targets: Novelty justification via convergence analysis; links algorithm structure to performance claims)*

10. **What empirical benchmarks (e.g., GridWorld, Chain MDPs) are standard for evaluating convergence speed in deterministic episodic Q-learning, and how do existing methods (Dyna-Q, MC, value iteration) perform on them relative to RBQL’s one-pass update?**  
*(Targets: Contextual benchmarking; prepares for experimental validation and comparison)*