# MONTE CARLO CONTROL in Easy21 (15 marks)
# tybens (10/12/20)

import enviro  # step(), first_draw()
import utils
import numpy as np


def monteCarlo(episodes, N_0):
    N_s_a = np.zeros(
        (10, 21, 2))  # number of times state s has been visited with action a [dealer_showing][psum][action]
    Q_s_a = np.zeros((10, 21, 2))
    for i in range(episodes):
        N_s_a, Q_s_a = episode(N_s_a, Q_s_a, N_0)

    # calculate optimal value function V*(s) = max_a(Q*(s, a))
    return Q_s_a


def episode(N_s_a, Q_s_a, N_0):
    state = [enviro.first_draw(), enviro.first_draw()]
    terminal = False
    while not terminal:
        index_action = choose_action(state, Q_s_a, N_s_a, N_0)  # 0 = stick, 1 = hit

        # increment Number of times action chosen given state 
        N_s_a[state[0] - 1][state[1] - 1][index_action] += 1

        # take a step
        ret = enviro.step(state, index_action)

        # increment mean value for action given state
        Q_s_a[state[0] - 1][state[1] - 1][index_action] += (1 / N_s_a[state[0] - 1][state[1] - 1][index_action]) * (
                ret[1] - Q_s_a[state[0] - 1][state[1] - 1][index_action])

        # update state
        state = ret[0]

        # is it terminal?
        terminal = ret[2]
    return N_s_a, Q_s_a


def choose_action(state, Q_s_a, N_s_a, N_0):
    # only usable within state space of N_s_a and Q_s_a
    index_dealer = state[0] - 1
    index_psum = state[1] - 1
    epsilon = N_0 / (N_0 + sum(N_s_a[index_dealer][index_psum]))
    greedy_choice = np.argmax(Q_s_a[index_dealer][index_psum])
    if greedy_choice == 0:
        ret = np.random.choice([0, 1], p=[(epsilon / 2 + 1 - epsilon), epsilon / 2])  # 0 = stick, 1 = hit
    elif greedy_choice == 1:
        ret = np.random.choice([0, 1], p=[epsilon / 2, (epsilon / 2 + 1 - epsilon)])  # 0 = stick, 1 = hit
    return ret


episodes = int(1e6)
N_0 = 100

Q = monteCarlo(episodes, N_0)

import dill as pickle

pickle.dump(Q, open('Q.dill', 'wb'))

# GRAPHING

trueQ = pickle.load(open('Q.dill', 'rb'))

utils.plotVstar(trueQ)

utils.plotVstarHeatMap(trueQ)

title = 'optimal policy using MC after 1,000,000 episodes'
utils.plotOptimalPolicy(trueQ, title)
