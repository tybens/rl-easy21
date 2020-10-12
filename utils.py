import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set()


def plotVstar(Q):
    # defining surface and axes
    x = np.array(range(10))
    y = np.array(range(21))
    X, Y = np.meshgrid(y, x)
    Z = np.maximum(Q[:, :, 0], Q[:, :, 1])

    fig_dims = (20, 12)
    fig = plt.figure(figsize=fig_dims)

    # syntax for 3-D plotting 
    ax = plt.axes(projection='3d')

    # syntax for plotting 
    ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='green', linewidths=0)
    ax.set_title('Optimal Value At Each State')
    plt.xlabel('Player Sum')
    plt.ylabel('Dealer Showing')
    ax.set_facecolor('white')
    ax.view_init(elev=50, azim=200)
    plt.show()


def plotVstarHeatMap(Q):
    V_star_s = np.maximum(Q[:, :, 0], Q[:, :, 1])

    fig_dims = (20, 12)
    fig, ax = plt.subplots(figsize=fig_dims)

    sns.heatmap(ax=ax, data=V_star_s, center=0, annot=True)
    plt.xlabel('Player Sum')
    plt.ylabel('Dealer Showing');


def plotOptimalPolicy(Q, title):
    optimal_policy = np.argmax(Q, axis=2)

    fig_dims = (20, 12)
    fig, ax = plt.subplots(figsize=fig_dims)

    sns.heatmap(ax=ax, data=optimal_policy, annot=True)
    plt.xlabel('Player Sum')
    plt.ylabel('Dealer Showing')
    cbar = ax.collections[0].colorbar
    ax.set_xticks([i for i in range(1, 22)])
    ax.set_title(title, size=25)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['stick', 'hit'])


def plotMseEpisodes(all_MSEs, lambdas, title):
    # --- example data initialization ---
    # all_MSEs = list()
    # lambdas = [0, 1]
    # for lam in lambdas:
    #     all_MSEs.append(TD_Learning(episodes, N_0, gamma, lam, remember_every_MSE))
    # title = 'MSE over course of learning for TD Learning using Sarsa(lambda = {0, 1})'

    x = range(1, len(all_MSEs[0]) + 1)
    fig_dims = (20, 12)
    fig, ax = plt.subplots(1, figsize=fig_dims)
    for i, MSEs in enumerate(all_MSEs):
        plt.scatter(x, MSEs, label='lambda = {}'.format(lambdas[i]))

    ax.grid(True, c='k')
    plt.legend(fontsize=23, loc=2)
    plt.xlabel('Episode #', size=30)
    plt.ylabel('Mean Squared Error', size=30)
    ax.set_facecolor('white')
    ax.set_title(title, size=25)
    ax.spines['bottom'].set_color('k')
    ax.spines['left'].set_color('k')
    ax.spines['bottom'].set_linewidth(5)
    ax.spines['left'].set_linewidth(5)


def plotMseLambdas(lambdas, MSE_for_each_lambda, title):
    # --- example data initialization ---
    # lambdas = np.arange(0, 1.1, 0.1)
    # MSE_for_each_lambda = [((TD_Learning(episodes, N_0, gamma, lam, remember_every_MSE) - Q_star_s_a)**2).mean(axis=None) for lam in lambdas]
    # title = 'TD Learning comparing variable lambda in Sarsa(lambda)'
    fig_dims = (20, 12)
    fig, ax = plt.subplots(figsize=fig_dims)

    plt.scatter(lambdas, MSE_for_each_lambda, c='r', s=200)
    ax.set_facecolor('white')
    plt.xlabel('Lambda', size=30)
    plt.ylabel('Mean Squared Error', size=30)
    ax.set_title(title, size=25)
    ax.spines['bottom'].set_color('k')
    ax.spines['left'].set_color('k')
    ax.spines['bottom'].set_linewidth(5)
    ax.spines['left'].set_linewidth(5)
    ax.grid(True, c='k')
