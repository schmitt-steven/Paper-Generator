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

# Q-table
q_values = np.random.rand(num_of_states, num_of_actions) / 1000.0

# ------------------------------------------------------------------
# BACKWARD STATE GRAPH (Efficient predecessor lookup)
class BackwardStateGraph:
    def __init__(self):
        # Map: next_state -> list of (state, action) tuples that lead to it
        self.incoming = defaultdict(list)

    def add_transition(self, state, action, next_state):
        self.incoming[next_state].append((state, action))

    def get_predecessors(self, next_state):
        """Returns list of (state, action) that lead to `next_state`."""
        return self.incoming[next_state]

    def clear(self):
        self.incoming.clear()

# Instantiate the graph
backward_graph = BackwardStateGraph()

# ------------------------------------------------------------------
# EPISODE MEMORY — Now only stores the terminal state and reward!
episode_terminal_state = None
episode_terminal_reward = 0

def remember_transition(state, action, next_state):
    """Store transition in backward_graph. No duplicates needed."""
    backward_graph.add_transition(state, action, next_state)

def propagate_reward(reward):
    """Breadth-first backward propagation of reward through the graph."""
    global q_values, gamma
    updated = set()  # Track (state, action) pairs already updated
    queue = deque()

    # Start BFS from the terminal state (goal)
    queue.append((episode_terminal_state, 0))  # (state, level)
    
    while queue:
        current_state, level = queue.popleft()
        
        # Get all predecessors (state, action) that lead to current_state
        for state, action in backward_graph.get_predecessors(current_state):
            key = (state, action)
            
            if key not in updated:
                # Apply discounted reward: gamma^level * reward
                discount = gamma ** level
                q_values[state][action] = discount * reward  # You could also assign instead of += if desired
                updated.add(key)
                
                # Enqueue this predecessor for next level
                queue.append((state, level + 1))

    # Clear graph after propagation
    backward_graph.clear()

# ------------------------------------------------------------------
# HELPER FUNCTIONS (unchanged from original script)

def getAction(state):          # -1 for left, +1 for right
    global epsilon, Q
    if np.random.rand() <= epsilon:
        return random.choice([-1, 1])
    return (np.argmax(q_values[int(state)]) * 2) - 1

def getState(x_ball, y_ball, vx_ball, vy_ball, x_racket):
    return int((((x_ball * 13 + y_ball) * 2 + (vx_ball + 1) // 2)
                * 2 + (vy_ball + 1) // 2) * 12 + x_racket)

# ------------------------------------------------------------------
# INITIALISE PYGAME

pyg.init()
screen = pyg.display.set_mode((240, 260))
pygame_font = pyg.font.SysFont("arial", 15)

x_racket, x_ball, y_ball, vx_ball, vy_ball, score = 5, 1, 1, 1, 1, 0
episode = 0

file = open('reward_ODQN.txt', 'w')
clock = pyg.time.Clock()
cont = True

while cont:
    for event in pyg.event.get():
        if event.type == pyg.QUIT:
            cont = False

    # decay epsilon
    epsilon -= 1/400
    if epsilon < 0: epsilon = 0

    screen.fill((0, 0, 0))
    t = pygame_font.render(f"Score:{score} Episode:{episode}", True, (255, 255, 255))
    screen.blit(t, t.get_rect(centerx=screen.get_rect().centerx))

    # draw objects
    pyg.draw.rect(screen, (0, 128, 255), pyg.Rect(x_racket*20, 250, 80, 10))
    pyg.draw.rect(screen, (255, 100, 0),   pyg.Rect(x_ball*20, y_ball*20, 20, 20))

    # ---- PLAY ONE STEP ------------------------------------------
    state = getState(x_ball, y_ball, vx_ball, vy_ball, x_racket)
    action = getAction(state)                     # -1 or +1
    next_action_index = (action + 1)//2            # 0 or 1

    # perform action
    x_racket += action
    x_racket = max(0, min(11, x_racket))

    # ball movement
    x_ball += vx_ball
    y_ball += vy_ball
    if x_ball > 10 or x_ball < 1: vx_ball *= -1
    if y_ball > 11 or y_ball < 1: vy_ball *= -1

    reward = 0
    next_state = getState(x_ball, y_ball, vx_ball, vy_ball, x_racket)

    # remember the transition in backward_graph (no dedup needed)
    remember_transition(state, next_action_index, next_state)

    if y_ball == 12:  # terminal state reached
        reward = -1
        if x_ball >= x_racket and x_ball <= x_racket + 4:
            reward = +1

        episode += 1
        score += reward

        # --- CRITICAL: Propagate reward BACKWARDS using BFS ---
        episode_terminal_state = next_state  # the terminal state
        episode_terminal_reward = reward   # store it
        propagate_reward(reward)           # BFS backward propagation!

        file.write(f"{reward},")
        file.flush()

    # ---------------------------------------------------------------
    pyg.display.flip()
    clock.tick(60)          # 60 FPS

file.close()
pyg.quit()
