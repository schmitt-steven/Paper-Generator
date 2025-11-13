import pygame as pyg
import numpy as np
import random
from collections import defaultdict, deque

# ------------------------------------------------------------------
# PARAMETERS
epsilon = 1.0          # exploration rate (decays over time)
gamma   = 0.95         # discount factor
num_of_states  = 13*12*2*2*12
num_of_actions = 2

# Q-table (small initial values)
q_values = np.random.rand(num_of_states, num_of_actions) / 1000.0

# ------------------------------------------------------------------
# PERSISTENT MODEL (never cleared)
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

# Instantiate the persistent model
model = PersistentModel()

# ------------------------------------------------------------------
# RBQL UPDATE

def propagate_reward_rbql(terminal_state):
    """
    RBQL backward propagation with α=1 (equation 3 from paper).
    Q(s,a) = R(s,a) + γ * max(Q(next_state))
    """
    global q_values, gamma
    
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
            
            # RBQL update rule with α=1
            # Q(s,a) = R(s,a) + γ * max(Q(next_state))
            next_q = np.max(q_values[current_state])
            q_values[state][action_index] = reward + gamma * next_q
            
            updated.add(key)
            queue.append(state)

# ------------------------------------------------------------------
# HELPER FUNCTIONS

def getAction(state):
    """Return -1 (left) or +1 (right). Use epsilon-greedy on q_values."""
    global epsilon
    if np.random.rand() <= epsilon:
        return random.choice([-1, 1])
    # argmax returns 0 or 1 -> map to -1 or +1
    best_action_index = int(np.argmax(q_values[int(state)]))
    return -1 if best_action_index == 0 else 1

def getState(x_ball, y_ball, vx_ball, vy_ball, x_racket):
    """Deterministic mapping to discrete state index (must match num_of_states)."""
    return int((((x_ball * 13 + y_ball) * 2 + (vx_ball + 1) // 2)
                * 2 + (vy_ball + 1) // 2) * 12 + x_racket)

# ------------------------------------------------------------------
# INITIALISE PYGAME

pyg.init()
screen = pyg.display.set_mode((240, 260))
pygame_font = pyg.font.SysFont("arial", 15)

x_racket, x_ball, y_ball, vx_ball, vy_ball, score = 5, 1, 1, 1, 1, 0
episode = 0

file = open('reward_RBQL.txt', 'w')
clock = pyg.time.Clock()
cont = True

while cont:
    for event in pyg.event.get():
        if event.type == pyg.QUIT:
            cont = False

    # decay epsilon
    epsilon -= 1/400
    if epsilon < 0:
        epsilon = 0

    screen.fill((0, 0, 0))
    t = pygame_font.render(f"Score:{score} Episode:{episode}", True, (255, 255, 255))
    screen.blit(t, t.get_rect(centerx=screen.get_rect().centerx))

    # draw objects
    pyg.draw.rect(screen, (0, 128, 255), pyg.Rect(x_racket*20, 250, 80, 10))
    pyg.draw.rect(screen, (255, 100, 0),   pyg.Rect(x_ball*20, y_ball*20, 20, 20))

    # ---- PLAY ONE STEP ------------------------------------------
    state = getState(x_ball, y_ball, vx_ball, vy_ball, x_racket)
    action = getAction(state)                     # -1 or +1
    action_index = (action + 1)//2                # 0 or 1

    # perform action
    x_racket += action
    x_racket = max(0, min(11, x_racket))

    # ball movement
    x_ball += vx_ball
    y_ball += vy_ball
    if x_ball > 10 or x_ball < 1:
        vx_ball *= -1
    if y_ball > 11 or y_ball < 1:
        vy_ball *= -1

    next_state = getState(x_ball, y_ball, vx_ball, vy_ball, x_racket)
    
    # Default step reward
    reward = 0
    
    if y_ball == 12:  # terminal state reached
        # Terminal reward depends on racket overlap
        reward = -1
        if x_ball >= x_racket and x_ball <= x_racket + 4:
            reward = +1
        
        episode += 1
        score += reward
        
        # Store terminal transition with reward
        model.add_transition(state, action_index, next_state, reward)
        
        # RBQL: propagate backwards through entire model
        propagate_reward_rbql(next_state)
        
        file.write(f"{reward},")
        file.flush()
    else:
        # Store non-terminal transition
        model.add_transition(state, action_index, next_state, reward)

    # ---------------------------------------------------------------
    pyg.display.flip()
    clock.tick(60)          # 60 FPS

file.close()
pyg.quit()