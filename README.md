# Randomized Exploration for Non-Stationary Stochastic Linear Bandits - UAI 2020

The algorithms are described and theoretically analyzed in the following work available at: https://arxiv.org/pdf/1912.05695.pdf. 
```
Baekjin Kim and Ambuj Tewari. 
Randomized exploration for non-stationary stochastic linear bandits. 
In Proceedings of the 36th Annual Conference on Uncertainty in Artificial Intelligence, 2020.
```

If you use this code in your paper, please cite the above work.

The code is built with Python 3.7.1 and it requires Numpy and Scipy to run experiments and the algorithms themselves. 

Also, the code is built on top of sourcecode of the NeurIPS19 paper of Yoan Russac et al., which is available at https://github.com/YRussac/WeightedLinearBandits.

This experiment is a simulation study based on real dataset to show empirical performances of D-LinTS and D-RandLinUCB, which uses Criteo Attribution Modeling for Bidding Dataset ("criteo_attribution_dataset.tsv.gz") available at https://ailab.criteo.com/criteo-attribution-modeling-bidding-dataset.

Due to code error in implementing parameter changes, revised code and corresponding figures are added in "Experiment_revised" and "Figure_revised" folders.
