# adapted from timbmg
# tybens (10/12/20)

import numpy as np
import dill as pickle
import enviro  # step(), first_draw()
import utils  # graphing

N0 = 100
actions = [0, 1]


def reset():
    theta = np.random.randn(3 * 6 * 2, 1)
    wins = 0

    return theta, wins


trueQ = pickle.load(open('Q.dill', 'rb'))

# step size
alpha = 0.01

# exploration probability
epsilon = 0.05

episodes = int(1e4)
lmds = np.arange(0, 1.1, 0.1)

mselambdas = np.zeros((len(lmds), episodes))
finalMSE = np.zeros(len(lmds))


def epsilonGreedy(state):
    if np.random.random() < epsilon:
        # explore
        action = np.random.choice(actions)

    else:
        # exploit
        action = np.argmax([Q(state, a) for a in actions])

    return action


def choose_action(state):
    epsilon = 0.05
    greedy_choice = np.argmax([Q(state, action) for action in actions])
    if greedy_choice == 0:
        ret = np.random.choice([0, 1], p=[(epsilon / 2 + 1 - epsilon), epsilon / 2])  # 0 = stick, 1 = hit
    elif greedy_choice == 1:
        ret = np.random.choice([0, 1], p=[epsilon / 2, (epsilon / 2 + 1 - epsilon)])  # 0 = stick, 1 = hit
    return ret


def features(state, a):
    f = np.zeros(3 * 6 * 2)
    d = state[0]
    p = state[1]
    for fi, (lower, upper) in enumerate(zip(range(1, 8, 3), range(4, 11, 3))):
        f[fi] = (lower <= d <= upper)

    for fi, (lower, upper) in enumerate(zip(range(1, 17, 3), range(6, 22, 3)), start=3):
        f[fi] = (lower <= p <= upper)

    f[-2] = 1 if a == 0 else 0
    f[-1] = 1 if a == 1 else 0

    return f.reshape(1, -1)


def Q(state, a):
    return np.dot(features(state, a), theta)


allFeatures = np.zeros((10, 21, 2, 3 * 6 * 2))
for d in range(10):
    for p in range(21):
        for a in range(0, 2):
            allFeatures[d, p, a] = features([d + 1, p + 1], a)


def allQ():
    return np.dot(allFeatures.reshape(-1, 3 * 6 * 2), theta).reshape(-1)


for li, lmd in enumerate(lmds):

    theta, wins = reset()

    for episode in range(episodes):

        terminated = False
        E = np.zeros_like(theta)  # Eligibility Trace

        # inital state and first action

        state = [enviro.first_draw(), enviro.first_draw()]
        a = choose_action(state)

        # Sample Environment
        while not terminated:

            statePrime, r, terminated = enviro.step(state, a)

            if not terminated:
                aPrime = choose_action(statePrime)
                tdError = r + Q(statePrime, aPrime) - Q(state, a)
            else:
                tdError = r - Q(state, a)

            E = lmd * E + features(state, a).reshape(-1, 1)
            gradient = alpha * tdError * E
            theta = theta + gradient

            if not terminated:
                state, a = statePrime, aPrime

        # bookkeeping
        if r == 1:
            wins += 1

        mse = np.sum(np.square(allQ() - trueQ.ravel())) / (21 * 10 * 2)
        mselambdas[li, episode] = mse

        if episode % 1000 == 0 or episode + 1 == episodes:
            print("Lambda=%.1f Episode %06d, MSE %5.3f, Wins %.3f" % (lmd, episode, mse, wins / (episode + 1)))

    finalMSE[li] = mse
    print("Lambda=%.1f Episode %06d, MSE %5.3f, Wins %.3f" % (lmd, episode, mse, wins / (episode + 1)))
    print("--------")

# GRAPHING

all_MSEs = [mselambdas[0], mselambdas[1]]
lambdas = [0, 1]
title = 'Linear Function Approximation, MSE as a function of Lambdas in TD(lambda)'
utils.plotMseEpisodes(all_MSEs, lambdas, title)

lambdas = np.arange(0, 1.1, 0.1)
title = 'LFA, TD(lambda) MSE as a function of lambda in Sarsa(lambda)'
mselambdas.shape
utils.plotMseLambdas(lambdas, finalMSE, title)
