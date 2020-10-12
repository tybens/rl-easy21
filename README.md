# Reinforcement Learning on Easy21

Reinforcement Learning is applied to Easy21. This is an assignment as part of David Silver's Reinforcement Learning Course at UCL. The assignment can be found [here](https://deepmind.com/learning-resources/-introduction-reinforcement-learning-david-silver).

## Monte-Carlo Control

`python3 monteCarlo.py`

The agent played 1 Million games (episodes) to obtain the following Value function:  
<img src="https://github.com/tybens/rl-easy21/blob/main/figures/mc_vstar.png" width="1000">

Visualized as a heatmap:
<img src="https://github.com/tybens/rl-easy21/blob/main/figures/mc_vstar_heatmap.png" width="1000">

## TD Learning

`python3 temporalDifference.py`

The MSE of Q, the state-action function, over the course of episodic learning. For each lambda, 10,000 Episodes have been measured against the Monte-Carlo 1 Million state-action function, saved in `Q.dill`:
<img src="https://github.com/tybens/rl-easy21/blob/main/figures/td_episodes.png" width="1000">

Mean Squared Error after 1,000 episodes for different lambdas:
<img src="https://github.com/tybens/rl-easy21/blob/main/figures/td_lambdas.png" width="1000">

The optimal policy as derived from 10,000 episodes of TD(lambda = 0.3):
<img src="https://github.com/tybens/rl-easy21/blob/main/figures/td_optimal.png" width="1000">

## Linear Function Approximation

`python3 lfa.py`

The matrix lookup-table approach of the previous models are replaced by coarse coding function approximator. This reduces the 420 state-action combinations to 36.

<img src="https://github.com/tybens/rl-easy21/blob/main/figures/lfa_episodes.png" width="1000">
  
<img src="https://github.com/tybens/rl-easy21/blob/main/figures/lfa_lambdas.png" width="1000">
