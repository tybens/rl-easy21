# IMPLEMENTATION OF Easy21 (10 marks)
import numpy as np


def step(s, a):
    # state s (s[0] = dealer card, s[1] = player sum)
    # action (hit or stick)
    # 
    psum = s[1]
    dealer = s[0]

    reward = 0
    is_terminal = False
    if a == 1:
        psum += draw_card()
        if psum > 21 or psum < 1:
            # player bust
            reward = -1
            is_terminal = True
    elif a == 0:
        while 17 > dealer > 0:
            dealer += draw_card()
        if dealer > 21 or dealer < 1 or psum > dealer:
            # dealer bust or player win
            reward = 1
        elif dealer > psum:
            reward = -1
        is_terminal = True
    else:
        print("Not a valid action: {} | should be 0 for stick, or 1 for hit".format(a))

    s_prime = [dealer, psum]

    return s_prime, reward, is_terminal


def draw_card():
    # sample card between [-10, 0) U (0, 10]
    card_drawn = (np.random.choice(10) + 1) * np.random.choice([-1, 1], p=[1 / 3, 2 / 3])
    return card_drawn


def first_draw():
    # sample card between (0, 10]
    card_drawn = np.random.choice(10) + 1
    return card_drawn
