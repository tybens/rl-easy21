# Sarsa(lambda) (15 marks)
# tybens (10/12/20)

import enviro  # step(), first_draw()
import utils  # graphing
import dill as pickle
import numpy as np

trueQ = pickle.load(open('Q.dill', 'rb'))


def choose_action(state, Q_s_a, N_s_a, N_0):
    # only usable within state space of N_s_a and Q_s_a
    index_dealer = state[0] - 1
    index_psum = state[1] - 1
    epsilon = N_0 / (N_0 + sum(N_s_a[index_dealer][index_psum]))
    greedy_choice = np.argmax(Q_s_a[index_dealer][index_psum])
    if greedy_choice == 0:
        ret = np.random.choice([0, 1], p=[(epsilon / 2 + 1 - epsilon), epsilon / 2])  # 0 = stick, 1 = hit
    else:
        ret = np.random.choice([0, 1], p=[epsilon / 2, (epsilon / 2 + 1 - epsilon)])  # 0 = stick, 1 = hit
    return ret


def TD_Learning(episodes, N_0, gamma, lam, remember_every_MSE):
    # pseudo code as comments:
    # initialize Q(s, a) arbitrarily (zero as specified), for all s in S and all a in A.
    Q_s_a = np.zeros((10, 21, 2))
    N_s_a = np.zeros((10, 21, 2))

    all_MSEs = []
    # repeat (for each episode):
    for _ in range(episodes):
        # initialize Eligibility trace for each s in S and a in A(s)
        E_trace = np.zeros((10, 21, 2))

        # initialize S, A
        state = [enviro.first_draw(), enviro.first_draw()]
        action = choose_action(state, Q_s_a, N_s_a, N_0)

        terminal = False
        # repeat for each step
        while not terminal:
            # state = [dealer, psum]
            # action = 0 or 1 (0 for stick, 1 for hit)

            # increment Number of times action chosen given state (for eps-greedy!)
            N_s_a[state[0] - 1][state[1] - 1][action] += 1

            # take action A, observe R, S'
            state_prime, R, terminal = enviro.step(state, action)

            if terminal:
                # terminal states can't choose action, so Q_s_a(S', *) is 0 when S' == terminal state
                prediction = 0
                TD_error = R + prediction - Q_s_a[state[0] - 1][state[1] - 1][action]
            else:
                # Choose A' from S' using policy derived from Q (eps-greedy)
                action_prime = choose_action(state_prime, Q_s_a, N_s_a, N_0)
                prediction = gamma * Q_s_a[state_prime[0] - 1][state_prime[1] - 1][action_prime]
                TD_error = R + prediction - Q_s_a[state[0] - 1][state[1] - 1][action]

            alpha = 1 / N_s_a[state[0] - 1][state[1] - 1][action]

            E_trace[state[0] - 1][state[1] - 1][action] += 1

            # For all s in S, a in A(s)
            # update Q based on E_trace
            Q_s_a = np.add(Q_s_a, alpha * TD_error * E_trace)
            # degrade E_trace
            E_trace = lam * gamma * E_trace

            state = state_prime
            if not terminal:
                action = action_prime

        if remember_every_MSE:
            all_MSEs.append(((Q_s_a - trueQ) ** 2).mean(axis=None))

    V_s = np.maximum(Q_s_a[:, :, 0], Q_s_a[:, :, 1])  # don't get to call it v_star...
    if remember_every_MSE:
        return all_MSEs
    return Q_s_a


# GRAPHING ---


episodes = 1000
N_0 = 100
gamma = 1
lam = 0.5
remember_every_MSE = False

lambdas = np.arange(0, 1.1, 0.1)
MSE_for_each_lambda = [((TD_Learning(episodes, N_0, gamma, lam, remember_every_MSE) - trueQ) ** 2).mean(axis=None) for
                       lam in lambdas]
title = 'TD Learning comparing variable lambda in Sarsa(lambda)'

utils.plotMseLambdas(lambdas, MSE_for_each_lambda, title)

episodes = 10000
remember_every_MSE = True
all_MSEs = list()
lambdas = [0, 1]
for lam in lambdas:
    all_MSEs.append(TD_Learning(episodes, N_0, gamma, lam, remember_every_MSE))
title = 'MSE over course of learning for TD Learning using Sarsa(lambda = {0, 1})'

utils.plotMseEpisodes(all_MSEs, lambdas, title)

lam = 0.3
remember_every_MSE = False
Q_s_a = TD_Learning(episodes, N_0, gamma, lam, remember_every_MSE)
title = 'optimal policy at lambda = {}'.format(lam)
utils.plotOptimalPolicy(Q_s_a, title)
