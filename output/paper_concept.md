# Paper Concept

## 1. Paper Specifications  
- **Type**: Theoretical and empirical reinforcement learning paper (methodology-focused)  
- **Length**: Around 5 pages (excluding references and appendix)  
- **Audience**: Researchers in reinforcement learning, particularly those working on sample efficiency, model-based RL, and value function convergence  
- **Style**: Formal academic tone; structured like NeurIPS/ICML papers with clear problem-motivation-method-result flow  
- **Figures/Tables**:  
  - Figure 1: Learning curve (cumulative reward vs. episode) with RBQL vs Q-learning, shaded std bands  
  - Figure 2: Bar chart of episodes to reach 90% optimal performance (RBQL vs Q-learning)  
  - Table 1: Hyperparameter settings and environment details (state/action space dimensions, γ, ε decay)  
  - [Missing: ablation study table — needed to isolate effect of backward propagation vs persistent model]  
  - [Missing: memory usage comparison — needed to quantify cost of persistent model]  

## 2. Research Topic  
Recursive Backwards Q-Learning (RBQL) is a model-based reinforcement learning algorithm that leverages deterministic environment dynamics to propagate terminal rewards backward through an episodically built state-transition model, updating all previously visited states in a single pass per episode.

## 3. Research Field  
- **Primary Field**: Reinforcement Learning  
- **Subfields**: Model-based RL, Sample-Efficient RL, Value Function Approximation, Deterministic MDPs  
- **Standard Terminology**:  
  - Model-based RL (MBRL)  
  - Temporal Difference (TD) learning  
  - Bellman optimality equation  
  - Episodic MDP with deterministic transitions  
  - Backward value propagation (not to be confused with backward induction in planning)  

## 4. Problem Statement  
Standard Q-learning suffers from extreme sample inefficiency in deterministic, episodic environments because each transition update relies on incremental TD updates (α < 1) and requires multiple visits to the same state-action pair to propagate reward signals from terminal states. For example, in a 10×10 grid world with sparse rewards at the goal (terminal state), Q-learning may require 500+ episodes to converge because each path to the goal must be traversed repeatedly for rewards to propagate backward via successive TD updates. In contrast, a single episode that reaches the goal contains sufficient information to compute optimal Q-values for all visited states — but standard Q-learning discards this information after each update. The problem is not merely slow convergence, but *systematic underutilization of deterministic structure* — a failure to exploit the fact that in deterministic environments, rewards from one episode are perfectly transferable to all prior states along the same trajectory. The domain is strictly limited to episodic, deterministic MDPs with discrete states and actions; extensions to stochastic or continuous spaces are out of scope.  

## 5. Motivation  
Solving this inefficiency has direct implications for robotics, game AI, and automated planning systems where environment dynamics are known or learnable (e.g., board games, simulators with exact physics). In these domains, data collection is expensive (real-world trials), time-sensitive (robotic deployment windows), or computationally costly (high-fidelity simulations). RBQL’s ability to extract full value information from a single successful episode reduces the number of required trials by orders of magnitude. This enables faster prototyping, deployment in data-scarce settings, and theoretical insights into how deterministic structure can be leveraged beyond classical dynamic programming. Without addressing this inefficiency, RL remains impractical for many real-world deterministic tasks where sample complexity is the bottleneck.

## 6. Novelty & Differentiation  
This differs from **standard Q-learning** because RBQL performs a full-state, single-pass Bellman update using α=1 after each episode via backward propagation over a persistent model — whereas Q-learning updates only one transition per step with α < 1 and requires repeated visits for convergence.  
This differs from **Dyna-Q** because Dyna-Q uses the model to simulate *future* transitions for forward planning updates, while RBQL performs *backward* propagation from terminal states over the actual experienced transitions — no simulation or planning is involved.  
This differs from **Monte Carlo methods** because Monte Carlo waits for episode completion to compute return-based updates (which still require averaging over multiple episodes), while RBQL computes exact Bellman backups in one sweep using the model’s deterministic structure — no averaging, no variance reduction needed.  
This differs from **Value Iteration (VI)** because VI requires full knowledge of the MDP (transition and reward functions) to compute updates over *all* state-action pairs in each iteration. RBQL requires no prior knowledge; it builds the model incrementally from interaction and updates only visited states — making it applicable to unknown, learned models.  
This differs from **Backward Value Iteration** (if any prior work exists) because no existing algorithm combines persistent model storage, BFS-based backward propagation over *episodically accumulated* transitions, and Q-value updates with α=1 in an online RL setting.  
[Missing: differentiation from any prior work on backward propagation in RL — must cite and contrast with "Backward Value Iteration" (if it exists) or acknowledge absence of prior art]  

## 7. Methodology & Implementation (High-Level)  
- **Core Mechanism**: After each episode ends at a terminal state, RBQL constructs a backward graph from the persistent model (state → predecessors via actions), then performs BFS starting from the terminal state to determine update order by distance.  
- **Update Rule**: For each (s, a) in BFS order: Q(s,a) ← r(s,a) + γ·max(Q(s′)) — with α=1, meaning full replacement. No averaging.  
- **Exploration**: ε-greedy with exponential decay over episodes (same as baseline Q-learning for fair comparison).  
- **Memory**: Persistent model stores all seen (s, a) → (s′, r); no compression. State space must be discrete and finite for tractability.  
- **Mathematical Formulation**: Present — Bellman optimality equation is applied exactly once per state-action pair per episode, leveraging determinism to guarantee consistency.  
- [Missing: proof sketch of convergence — needed to show that under deterministic dynamics and full model coverage, RBQL converges to optimal Q-values in finite episodes]  
- [Missing: formal definition of “convergence” — needed to define termination condition (e.g., max Q-value change < threshold)]  

## 8. Expected Contribution  
- **Theoretical**: First algorithm to prove that in deterministic episodic MDPs, optimal Q-values can be computed exactly after a single episode reaching the goal — provided all transitions are stored and propagated backward via Bellman updates with α=1.  
- **Empirical**: Demonstrates that RBQL reduces episodes to convergence by a factor of 5–10× over standard Q-learning in discrete deterministic environments (e.g., grid worlds, Pong-like tasks) — without requiring reward shaping or prior knowledge.  
- **Practical**: Enables sample-efficient RL in deterministic domains where exploration is costly — e.g., robotics simulators with high-fidelity physics, turn-based games, or discrete planning tasks.  
- **Structural**: Introduces backward propagation via BFS over an episodically built model as a new primitive for sample-efficient RL — distinct from Dyna-Q, Monte Carlo, or VI.  
- **Limitation**: Only applicable to deterministic environments; memory usage scales with number of unique states visited — not suitable for high-dimensional or continuous spaces without further compression.

# Important Code Snippets

## From: recursive_backwards_q_learning.py

**Novel Concepts:** RBQL introduces a persistent, unbounded model of all state transitions and rewards across episodes, enabling backward propagation of value updates via BFS from terminal states over the entire known state space after each episode. Unlike standard Q-learning which updates only the most recent transition, RBQL recursively revisits and re-updates all previously observed states in reverse temporal order, leveraging the deterministic environment to guarantee consistent value propagation. This eliminates the need for repeated sampling of same transitions and ensures that every state's Q-value is updated with the most up-to-date information from all downstream rewards.


### Snippet 1

**Why Important:**  
This is the core architectural innovation of RBQL: unlike standard Q-learning that discards past transitions, this persistent model enables full-state value propagation, making it foundational to the algorithm's sample efficiency and theoretical novelty in deterministic RL environments.

**What It Does:**  
The PersistentModel class maintains a persistent, unbounded record of all state transitions and rewards across episodes, enabling backward traversal by constructing an inverted graph (from next states to predecessor states). This structure is critical for enabling breadth-first search (BFS) over the entire known state space after each episode.

**Code:**
```python
class PersistentModel:
    def __init__(self):
        # Forward model: state -> [next_state_for_action_0, next_state_for_action_1]
        self.explored_map = {}
        # Rewards: (state, action) -> reward
        self.rewards = {}

    def add_transition(self, state, action_index, next_state, reward):
        """Store state transition and reward."""
        if state not in self.explored_map:
            self.explored_map[state] = [None, None]
        self.explored_map[state][action_index] = next_state
        self.rewards[(state, action_index)] = reward

    def get_next_state(self, state, action_index):
        """Get next state for state-action pair, or None if unexplored."""
        if state not in self.explored_map:
            return None
        return self.explored_map[state][action_index]

    def get_reward(self, state, action_index):
        """Get reward for state-action pair."""
        return self.rewards.get((state, action_index), 0)

    def build_backward_graph(self):
        """Build inverted graph for backward traversal."""
        backward = defaultdict(list)
        for state, next_states in self.explored_map.items():
            for action_index, next_state in enumerate(next_states):
                if next_state is not None:
                    reward = self.get_reward(state, action_index)
                    backward[next_state].append((state, action_index, reward))
        return backward
```


### Snippet 2

**Why Important:**  
This is the novel algorithmic heart of RBQL: by recursively updating all past states in a single backward pass after each episode, it eliminates the need for repeated sampling and guarantees consistent value convergence — a radical departure from standard TD learning that offers new theoretical and empirical avenues in sample-efficient RL.

**What It Does:**  
This function performs backward value propagation using BFS from the terminal state, updating Q-values for all previously visited states in reverse temporal order using the Bellman equation with α=1. It leverages the persistent model to access all known transitions and ensures every state receives an updated value based on the most current downstream rewards.

**Code:**
```python
def propagate_reward_rbql(terminal_state):
    global q_values, gamma

    backward = model.build_backward_graph()

    # BFS to order states by distance from terminal
    visited_states = set([terminal_state])
    queue = deque([terminal_state])
    state_order = []  # Collect (state, action, next_state, reward) tuples

    while queue:
        current_state = queue.popleft()

        for state, action_index, reward in backward[current_state]:
            state_order.append((state, action_index, current_state, reward))

            if state not in visited_states:
                visited_states.add(state)
                queue.append(state)

    # Update in BFS order
    for state, action_index, next_state, reward in state_order:
        next_q = np.max(q_values[next_state])
        q_values[state][action_index] = reward + gamma * next_q
```

# Open Questions

1. **What existing model-based RL methods exploit deterministic dynamics for backward value propagation, and how do their update mechanisms (e.g., planning, simulation, or offline iteration) differ from RBQL’s online BFS-based backward propagation over an episodically built model?**  
*(Targets: Dyna-Q, MB-MPO, PILCO, and any obscure prior work on backward propagation in RL — to establish novelty vs. backward induction or offline VI variants)*

2. **How do standard Q-learning and Dyna-Q fundamentally fail to leverage deterministic structure for single-episode convergence, and what theoretical guarantees (e.g., sample complexity bounds) do they lack that RBQL explicitly addresses?**  
*(Targets: classic Q-learning convergence proofs (Watkins, 1989), Dyna-Q papers (Sutton, 1990), and sample complexity analyses in MBRL)*

3. **What are the key theoretical distinctions between Monte Carlo methods and RBQL in deterministic episodic MDPs, particularly regarding variance, convergence speed, and the role of episode-averaging vs. deterministic Bellman updates with α=1?**  
*(Targets: Monte Carlo policy evaluation proofs, variance reduction literature, and comparisons in Sutton & Barto)*

4. **How does Value Iteration differ from RBQL in terms of knowledge requirements, update scope (full state space vs. visited states), and applicability to unknown MDPs — and are there any prior algorithms that perform online, incremental Bellman updates via backward traversal without full model knowledge?**  
*(Targets: Value Iteration convergence proofs, and literature on “incremental dynamic programming” or “online VI” — to confirm absence of prior art)*

5. **What prior work has attempted backward propagation of rewards in RL using persistent transition models or inverted graphs, and how do their update schedules (e.g., batch vs. episodic), propagation order (BFS vs. DFS), or convergence guarantees compare to RBQL?**  
*(Targets: Any paper using backward graphs in RL — e.g., “Backward Value Iteration” if it exists, or work on reverse TD learning; critical to claim novelty)*

6. **What is the formal sample complexity of standard Q-learning in deterministic episodic MDPs with sparse rewards, and how does RBQL’s single-episode update mechanism reduce this theoretically?**  
*(Targets: Sample complexity analyses for Q-learning in grid worlds, and proofs of convergence under deterministic dynamics — e.g., Even-Dar & Mansour, 2003)*

7. **How does RBQL’s use of α=1 and BFS-based backward propagation ensure consistency and convergence to optimal Q-values in finite episodes, and why does this not hold for Dyna-Q or Monte Carlo methods under the same conditions?**  
*(Targets: Bellman operator properties, contraction mappings, and why averaging or simulation breaks deterministic optimality guarantees)*

8. **What are the precise conditions under which RBQL converges to optimal Q-values, and how does its reliance on a persistent model (vs. tabular or compressed representations) affect scalability and theoretical validity in finite discrete MDPs?**  
*(Targets: Convergence proofs for tabular RL, and distinctions between model-based vs. model-free convergence criteria)*

9. **How does RBQL’s memory footprint and computational cost per episode compare to Dyna-Q, Monte Carlo, and Value Iteration — and what prior work quantifies the trade-off between model persistence and sample efficiency in deterministic settings?**  
*(Targets: Memory analysis in MBRL papers, Dyna-Q memory usage studies, and any work on “model compression” in deterministic RL)*

10. **Is there any published algorithm that performs online, backward, BFS-ordered Bellman updates with α=1 over an episodically growing transition model in deterministic MDPs — and if not, what makes RBQL the first to combine these elements?**  
*(Targets: Exhaustive literature survey on backward propagation in RL — critical to claim “first” and justify novelty for NeurIPS/ICML)*