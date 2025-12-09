"""
Basic Q-Learning
- Updates only the current transition on each step
- Update: Q(s,a) = Q(s,a) + α * (R + γ * max(Q(s')) - Q(s,a))
- Slower convergence but simpler
"""

import pygame as pyg
import numpy as np
import random

# ------------------------------------------------------------------
# PARAMETERS
epsilon = 1.0          # exploration rate (decays over time)
alpha   = 0.1          # learning rate
gamma   = 0.95         # discount factor
num_of_states  = 13*12*2*2*12
num_of_actions = 2

# Q-table (small initial values)
q_values = np.random.rand(num_of_states, num_of_actions) / 1000.0

# ------------------------------------------------------------------
# Q-LEARNING UPDATE

def update_q(state, action_index, reward, next_state):
    """Standard Q-learning update for single transition."""
    global q_values, alpha, gamma
    
    best_next_q = np.max(q_values[next_state])
    td_target = reward + gamma * best_next_q
    td_error = td_target - q_values[state][action_index]
    q_values[state][action_index] += alpha * td_error

# ------------------------------------------------------------------
# HELPER FUNCTIONS

def getAction(state):
    """Return -1 (left) or +1 (right). Use epsilon-greedy on q_values."""
    global epsilon
    if np.random.rand() <= epsilon:
        return random.choice([-1, 1])
    best_action_index = int(np.argmax(q_values[int(state)]))
    return -1 if best_action_index == 0 else 1

def getState(x_ball, y_ball, vx_ball, vy_ball, x_racket):
    """Deterministic mapping to discrete state index."""
    return int((((x_ball * 13 + y_ball) * 2 + (vx_ball + 1) // 2)
                * 2 + (vy_ball + 1) // 2) * 12 + x_racket)

# ------------------------------------------------------------------
# INITIALISE PYGAME

pyg.init()
screen = pyg.display.set_mode((240, 260))
pygame_font = pyg.font.SysFont("arial", 15)

x_racket, x_ball, y_ball, vx_ball, vy_ball, score = 5, 1, 1, 1, 1, 0
episode = 0

file = open('reward_QLearning_basic.txt', 'w')
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
    action = getAction(state)
    action_index = (action + 1)//2

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
        reward = -1
        if x_ball >= x_racket and x_ball <= x_racket + 4:
            reward = +1
        
        episode += 1
        score += reward
        
        # Basic Q-learning: update only this transition
        update_q(state, action_index, reward, next_state)
        
        file.write(f"{reward},")
        file.flush()
    else:
        # Update non-terminal transition too
        update_q(state, action_index, reward, next_state)

    # ---------------------------------------------------------------
    pyg.display.flip()
    clock.tick(60)

file.close()
pyg.quit()