# Portfolio Optimization with Deep Bayesian Bandits

## Problem statement

```
A task where a fixed amount of resources must be allocated between competing resources 
that maximizes expected gain. 
Typically these problems involve an exploration/exploitation trade-off.
As exploration risks loss of money.

Multi-Armed bandits would be the ideal solution.
```

#### Implementation of Linear Full Posterior Bandits for portfolio optimization

## What does this do
- Walk forward Optimization to be done every few months to give :

    - x Training length
    - y Testing length
    - z Live trading length
    
- Training of model using walk forward opt results.
- Prediction of next best portfolio holdings ratios using

    - X training length for scope of data to train for.
    - y testing length for validation of training results
    - z Actual length in market where results of optimization
    are good for before having to optimize again.

< Insert Sample optimization output here>


This  corresponds to the *[Deep Bayesian Bandits Showdown: An Empirical
Comparison of Bayesian Deep Networks for Thompson
Sampling](https://arxiv.org/abs/1802.09127)* paper, published in
[ICLR](https://iclr.cc/) 2018. 

```
@article{riquelme2018deep, title={Deep Bayesian Bandits Showdown: An Empirical
Comparison of Bayesian Deep Networks for Thompson Sampling},
author={Riquelme, Carlos and Tucker, George and Snoek, Jasper},
journal={International Conference on Learning Representations, ICLR.}, year={2018}}
```

### Installation

    WIP
    
### Usage

    WIP

## Contextual Bandits

Contextual bandits are a rich decision-making framework where an algorithm has
to choose among a set of *k* actions at every time step *t*, after observing
a context (or side-information) denoted by *X<sub>t</sub>*. The general pseudocode for
the process if we use algorithm **A** is as follows:

```
At time t = 1, ..., T:
  1. Observe new context: X_t
  2. Choose action: a_t = A.action(X_t)
  3. Observe reward: r_t
  4. Update internal state of the algorithm: A.update((X_t, a_t, r_t))
```

The goal is to maximize the total sum of rewards: &sum;<sub>t</sub> r<sub>t</sub>

## Thompson Sampling

Thompson Sampling is a meta-algorithm that chooses an action for the contextual
bandit in a statistically efficient manner, simultaneously finding the best arm
while attempting to incur low cost. Informally speaking, we assume the expected
reward is given by some function
**E**[r<sub>t</sub> | X<sub>t</sub>, a<sub>t</sub>] = f(X<sub>t</sub>, a<sub>t</sub>).
Unfortunately, function **f** is unknown, as otherwise we could just choose the
action with highest expected value:
a<sub>t</sub><sup>*</sup> = arg max<sub>i</sub> f(X<sub>t</sub>, a<sub>t</sub>).

The idea behind Thompson Sampling is based on keeping a posterior distribution
&pi;<sub>t</sub> over functions in some family f &isin; F after observing the first
*t-1* datapoints. Then, at time *t*, we sample one potential explanation of
the underlying process: f<sub>t</sub> &sim; &pi;<sub>t</sub>, and act optimally (i.e., greedily)
*according to f<sub>t</sub>*. In other words, we choose
a<sub>t</sub> = arg max<sub>i</sub> f<sub>t</sub>(X<sub>t</sub>, a<sub>i</sub>).
Finally, we update our posterior distribution with the new collected
datapoint (X<sub>t</sub>, a<sub>t</sub>, r<sub>t</sub>).

The main issue is that keeping an updated posterior &pi;<sub>t</sub> (or, even,
sampling from it) is often intractable for highly parameterized models like deep
neural networks. The algorithms we list in the next section provide tractable
*approximations* that can be used in combination with Thompson Sampling to solve
the contextual bandit problem.
