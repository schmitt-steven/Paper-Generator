# Paper Concept: Recursive Backwards Q-Learning

## 1. Paper Specifications

- **Paper Type:** Conference research paper (ML/RL domain)
- **Length:** 5–8 pages
- **Audience:** Academic researchers in reinforcement learning, machine learning, and AI communities
- **Style:** Formal academic writing; must include citations, proofs or theoretical arguments where applicable
- **Figures/Tables:** Optional but encouraged if they enhance clarity or illustrate algorithmic behavior

## 2. Research Topic

The paper presents Recursive Backwards Q-Learning (RBQL) — a **model-based** reinforcement learning algorithm designed for deterministic, episodic environments with sparse rewards. Unlike conventional Q-learning that incrementally updates Q-values forward in time after each step, RBQL builds a persistent model of the environment and, at the end of each episode, performs a complete backward propagation from the terminal state through all known states. This approach uses direct value assignment (α=1) rather than incremental learning, enabling optimal value estimation in a single backward pass through the accumulated model. RBQL dramatically accelerates convergence in deterministic environments by leveraging the fact that state transitions are consistent and can be reliably modeled.

## 3. Research Field

- **Primary Field:** Reinforcement Learning (RL)
- **Subfields:**
  - Model-Based Reinforcement Learning
  - Temporal Difference Learning
  - Planning and Control in Sequential Decision Making
  - Deterministic Environment Learning
- **Standard Terminology:**
  - Q-learning, Bellman equation, reward propagation, temporal difference (TD), episodic tasks, state-action value function, model-based learning, state transition model

## 4. Problem Statement

Classical Q-learning and related TD methods are designed to be general-purpose, model-free algorithms. However, this generality comes at a cost when applied to **deterministic, episodic environments** with sparse rewards.

Key limitations include:
- **Slow reward propagation**: Terminal rewards must propagate backward through many episodes before influencing early-episode decisions
- **Wasted information**: In deterministic environments, the agent ignores readily available information about consistent state transitions
- **Incremental learning overhead**: Using learning rate α < 1 requires multiple visits to converge to optimal values, even when the environment dynamics are known
- **Inefficient credit assignment**: Each state-action pair is updated independently, requiring extensive exploration

Formally, the standard Q-learning update:

$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha [r_t + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t)]$$

requires multiple episode replays before terminal rewards properly influence early actions. With α = 0.1, it takes approximately 10 successful episodes for a terminal reward to propagate just one state backward at 90% strength.

## 5. Motivation

This research addresses the inefficiency of model-free methods in deterministic environments by introducing a model-based approach that leverages environment consistency. RBQL is particularly effective for:

- **Deterministic episodic tasks** where state transitions are consistent and predictable
- **Sparse reward environments** where only terminal states provide meaningful feedback
- **Path-finding and navigation problems** where the goal is to find optimal trajectories
- **Environments with long horizons** where credit assignment is challenging

Applications include:
- Grid world navigation and maze solving
- Deterministic game playing (e.g., puzzle games, certain Atari environments)
- Robot path planning in known, static environments
- Any sequential decision-making task where exploration reveals consistent dynamics

The key insight is that in deterministic environments, once you know how the world works, you can compute optimal values directly rather than learning them incrementally.

## 6. Novelty & Differentiation

RBQL differs from classical and contemporary RL methods as follows:

### Versus Standard Q-Learning:
- **Model-based vs. Model-free**: RBQL explicitly builds and maintains a state transition model (`explored_map`), while Q-learning operates without environmental knowledge
- **Direct assignment vs. Incremental learning**: RBQL uses α=1 for complete value replacement, while Q-learning uses α < 1 for gradual convergence
- **Batch updates vs. Online updates**: RBQL updates all known states after each episode; Q-learning updates one state-action pair per step
- **Backward propagation vs. Forward learning**: RBQL propagates values backward from terminal states; Q-learning learns forward through experience

### Versus SARSA / TD(λ):
- RBQL does not rely on eligibility traces or λ-return approximations
- Instead of gradual credit assignment, RBQL computes exact discounted returns through the model
- RBQL's backward traversal eliminates the need for trace decay mechanisms

### Versus Dynamic Programming:
- Unlike DP, RBQL does not require a priori knowledge of transition probabilities
- RBQL builds its model through exploration rather than starting with complete environment knowledge
- RBQL uses Q-values (state-action) rather than V-values (state-only)

### Versus Dyna-Q and Model-Based Methods:
- Dyna-Q interleaves planning and learning but still uses incremental updates
- RBQL performs complete value propagation through the entire model after each episode
- RBQL's backward propagation guarantees optimal values for all known states in deterministic environments

### Versus Monte Carlo Methods:
- MC methods also wait until episode end, but update only visited states
- RBQL updates ALL known states in the model, not just the current episode's trajectory
- RBQL uses the Bellman equation rather than sample returns

## 7. Methodology & Implementation (High-Level)

### 7.1 Persistent Model Construction

During exploration across ALL episodes, RBQL maintains:

**Forward Model (explored_map):**
$$\mathcal{M}: (s, a) \rightarrow s'$$

Stores the next state for each state-action pair.

**Reward Model:**
$$\mathcal{R}: (s, a) \rightarrow r$$

Stores the immediate reward for each state-action pair.

**Key Property:** This model persists across all episodes and continuously grows as new states are explored.

### 7.2 Backward Graph Construction

After each episode, RBQL inverts the forward model to create a backward graph:

$$\mathcal{G}_{back} = \{ s' \rightarrow [(s, a, r) : \mathcal{M}(s,a) = s'] \}$$

Each state maps to all (state, action, reward) triples that lead to it. This enables efficient predecessor lookup during backward propagation.

### 7.3 RBQL Update Rule with α=1

The key innovation is setting α=1 in the Q-learning update equation. This simplifies to:

$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + 1 \cdot [r_t + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t)]$$

$$Q(s_t, a_t) \leftarrow r_t + \gamma \max_{a'} Q(s_{t+1}, a')$$

This is a **direct assignment**, not incremental learning. The Q-value now depends solely on the immediate reward and the discounted value of the best next state.

**Why α=1 works in deterministic environments:**
- State transitions are consistent, so the model is accurate
- No need to average over multiple experiences
- Immediate convergence to correct values given accurate next-state values

### 7.4 Backward Propagation Algorithm

After terminal state $s_T$ is reached with reward $R_T$:

1. **Build backward graph** from persistent model
2. **Initialize BFS queue** with terminal state
3. **For each state in queue** (breadth-first order):
   - For each predecessor $(s, a, r)$ that leads to current state:
     - Compute: $Q(s,a) = r + \gamma \cdot \max_{a'} Q(\text{current\_state}, a')$
     - Add $s$ to queue for further backward propagation
4. **Continue until all reachable states processed**

**Key Properties:**
- Each state-action pair is updated exactly once per propagation
- BFS order ensures next-state values are already optimal when updating predecessors
- All known states benefit from each terminal reward, not just visited states

### 7.5 Pseudocode

```
Initialize Q(s,a) to small random values
Initialize explored_map = {}
Initialize rewards = {}

For each episode:
    While not terminal:
        Observe state s
        Choose action a (ε-greedy on Q)
        Execute action, observe s', r
        
        // Build persistent model
        explored_map[s][a] = s'
        rewards[s][a] = r
    
    // Backward propagation with α=1
    backward_graph = invert(explored_map)
    queue = [terminal_state]
    updated = {}
    
    While queue not empty:
        current_state = dequeue()
        
        For each (s, a, r) in backward_graph[current_state]:
            If (s,a) not in updated:
                Q(s,a) = r + γ * max_a' Q(current_state, a')
                updated.add((s,a))
                enqueue(s)
```

## 8. Expected Contribution

The expected contribution includes:

### Theoretical Contributions:
- **Optimal value convergence**: Proof that RBQL converges to optimal Q-values in deterministic environments after sufficient exploration
- **Sample efficiency analysis**: Demonstration that RBQL requires O(|S||A|) exploration steps vs. O(|S||A|/α) for standard Q-learning
- **Convergence rate comparison**: Formal analysis showing RBQL's exponential advantage in long-horizon tasks

### Empirical Contributions:
- **Dramatic speedup in deterministic environments**: 10-100x faster convergence on maze navigation tasks
- **Reduced variance**: More stable learning due to batch updates rather than single-sample updates
- **Scalability demonstration**: Successful application to large state spaces (50×50 grids)

### Algorithmic Contributions:
- **Novel backward propagation mechanism**: Efficient graph-based value propagation structure
- **Persistent model architecture**: Data structure design for incremental model building
- **Exploration strategy**: Integration of model-based exploration with value-based action selection

### Practical Contributions:
- **Clear applicability conditions**: Explicit characterization of when RBQL outperforms model-free methods
- **Implementation guidance**: Concrete algorithms and data structures for practitioners
- **Performance benchmarks**: Quantitative comparison against Q-learning, SARSA, and Dyna-Q


# Open Questions for Literature Search

1. **How do model-based RL methods like Dyna-Q, MBPO, and World Models handle the trade-off between model accuracy and planning efficiency in deterministic vs. stochastic environments?**
   - *Target*: Papers on model-based RL, model learning, and model-based planning

2. **What are the theoretical convergence guarantees for Q-learning with α=1 vs. α<1 in deterministic environments?**
   - *Target*: Theoretical RL papers on convergence properties, learning rates, and Bellman operators

3. **How do existing backward propagation or credit assignment methods in RL compare to RBQL's graph-based approach?**
   - *Target*: Literature on eligibility traces, n-step returns, Monte Carlo tree search, backwards induction

4. **What is the computational complexity of maintaining and inverting state transition graphs in large-scale RL problems?**
   - *Target*: Papers on graph algorithms in RL, memory-efficient representations, and scalability

5. **How does RBQL's approach compare to dynamic programming methods that also perform full-width backups?**
   - *Target*: Classic DP literature (Bellman, Howard), value iteration, policy iteration

6. **What exploration strategies are most effective for model-based RL in deterministic environments?**
   - *Target*: Papers on exploration bonuses, curiosity-driven learning, and optimism under uncertainty

7. **How do prioritized experience replay and other replay mechanisms in deep RL relate to RBQL's batch update strategy?**
   - *Target*: DQN variants, prioritized sweeping, experience replay literature

8. **What are the sample complexity bounds for learning in deterministic MDPs, and how does RBQL compare?**
   - *Target*: PAC-learning in RL, sample complexity analysis, optimal exploration

9. **How do heuristic search methods (A*, Dijkstra) used in planning relate to RBQL's backward value propagation?**
   - *Target*: AI planning literature, heuristic search, and connections to RL

10. **What are the practical considerations for implementing model-based RL in real-world deterministic systems (robotics, logistics, games)?**
    - *Target*: Applied RL papers, robotics literature, industrial applications

# Important Code Snippets

## From: recursive_backwards_q_learning.py

**Novel Concepts:** The code implements Recursive Backwards Q-Learning (RBQL), a model-based RL algorithm that builds a persistent state transition model and performs complete backward value propagation after each episode using direct assignment (α=1) rather than incremental learning.

### Snippet 1: Persistent Model Structure

**Why Important:**  
This data structure is the foundation of RBQL's model-based approach. Unlike standard Q-learning, RBQL explicitly maintains knowledge of state transitions and rewards across all episodes, enabling efficient backward propagation.

**What It Does:**  
Stores the forward model (state transitions) and reward function persistently across episodes. Provides methods to build an inverted backward graph for efficient predecessor lookup during value propagation.

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
    
    def build_backward_graph(self):
        """Build inverted graph for backward traversal."""
        backward = defaultdict(list)
        for state, next_states in self.explored_map.items():
            for action_index, next_state in enumerate(next_states):
                if next_state is not None:
                    reward = self.rewards[(state, action_index)]
                    backward[next_state].append((state, action_index, reward))
        return backward
```

### Snippet 2: RBQL Backward Propagation with α=1

**Why Important:**
This is the core RBQL algorithm. It performs complete backward value propagation through the entire accumulated model using direct assignment (α=1), enabling immediate convergence to optimal values in deterministic environments.

**What It Does:**
Starting from the terminal state, performs breadth-first traversal through the backward graph. For each predecessor state-action pair, directly assigns the optimal Q-value using the Bellman equation: Q(s,a) = r + γ * max(Q(s')). Each state-action pair is updated exactly once per propagation.

**Code:**
```python
def propagate_reward_rbql(terminal_state, q_values, model, gamma=0.95):
    """
    RBQL backward propagation with α=1 (direct assignment).
    Q(s,a) = R(s,a) + γ * max(Q(next_state))
    """
    # Build backward graph from persistent model
    backward = model.build_backward_graph()
    
    # BFS from terminal state
    updated = set()
    queue = deque([terminal_state])
    
    while queue:
        current_state = queue.popleft()
        
        # For each predecessor (s, a_index, r) that leads to current_state
        for state, action_index, reward in backward[current_state]:
            key = (state, action_index)
            if key in updated:
                continue
            
            # RBQL update rule with α=1 (direct assignment)
            # Q(s,a) = R(s,a) + γ * max(Q(next_state))
            next_q = np.max(q_values[current_state])
            q_values[state][action_index] = reward + gamma * next_q
            
            updated.add(key)
            queue.append(state)
    
    # Model persists - NO clearing!
```

### Snippet 3: Episode Structure

**Why Important:**
Shows how RBQL integrates model building during exploration with batch backward propagation at episode termination, contrasting with standard Q-learning's online updates.

**What It Does:**
During the episode, stores all transitions in the persistent model. When terminal state is reached, triggers complete backward propagation through entire accumulated model, not just current episode's trajectory.

**Code:**
```python
# During episode
while not terminal:
    state = get_current_state()
    action = epsilon_greedy(state, q_values)
    next_state, reward = environment.step(action)
    
    # Add to persistent model
    model.add_transition(state, action, next_state, reward)
    
    if is_terminal(next_state):
        # RBQL: propagate through ENTIRE model
        propagate_reward_rbql(next_state, q_values, model, gamma)
        # Model persists for next episode!
```