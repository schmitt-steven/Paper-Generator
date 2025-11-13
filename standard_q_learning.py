import pygame as pyg
import numpy as np
import random

# ------------------------------------------------------------------
# PARAMETERS
epsilon = 1.0          # exploration rate (decays over time)
gamma   = 0.95         # discount factor
alpha   = 0.1          # learning rate
num_of_states  = 13*12*2*2*12
num_of_actions = 2

# Q-table (small initial values)
q_values = np.random.rand(num_of_states, num_of_actions) / 1000.0

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

def update_q_table(state, action_index, reward, next_state, terminal):
    """Standard Q-learning update."""
    global q_values, alpha, gamma
    
    if terminal:
        # Terminal state has no future value
        target = reward
    else:
        # Q-learning: use max of next state's Q-values
        target = reward + gamma * np.max(q_values[next_state])
    
    # Update Q-value
    q_values[state][action_index] = (1 - alpha) * q_values[state][action_index] + alpha * target

# ------------------------------------------------------------------
# INITIALISE PYGAME

pyg.init()
screen = pyg.display.set_mode((240, 260))
pygame_font = pyg.font.SysFont("arial", 15)

x_racket, x_ball, y_ball, vx_ball, vy_ball, score = 5, 1, 1, 1, 1, 0
episode = 0

file = open('reward_QL.txt', 'w')
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

    reward = 0
    next_state = getState(x_ball, y_ball, vx_ball, vy_ball, x_racket)
    terminal = False

    if y_ball == 12:  # terminal state reached
        terminal = True
        # terminal reward depends on racket overlap
        reward = -1
        if x_ball >= x_racket and x_ball <= x_racket + 4:
            reward = +1

        episode += 1
        score += reward

        file.write(f"{reward},")
        file.flush()

    # Standard Q-learning update (immediate)
    update_q_table(state, action_index, reward, next_state, terminal)

    # ---------------------------------------------------------------
    pyg.display.flip()
    clock.tick(60)          # 60 FPS

file.close()
pyg.quit()