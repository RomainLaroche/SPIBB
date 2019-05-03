## Implementation of Safe Policy Improvement with Baseline Bootstrapping

This project can be used to reproduce the experiments presented in the submitted paper.

For a detailed description of this see the paper.

## Prerequisites

The project is implemented in Python 3.5 and requires *numpy* and *scipy*.

## Usage

We include the following:
- Libraries of the following algorithms:
	* basic RL,
	* SPIBB:
		+ Pi_b-SPIBB,
		+ Pi_{\leq b}-SPIBB,
	* HCPI:
		+ doubly-robust,
		+ importance_sampling,
		+ weighted_importance_sampling,
        + weighted_per_decision_IS,
        + per_decision_IS,
    * Robust MDP,
    * and Reward-adjusted MDP.
- Environments:
	* Gridworld environment,
	* Random MDPs environment.
- Gridworld experiment of Section 3.1. Run:
		python gridworld_main.py #name_of_experiment# #random_seed#
- Gridworld experiment with random behavioural policy of Section 3.2. Run: 
		python gridworld_random_behavioural_main.py #name_of_experiment# #random_seed#
- Random MDPs experiment of Section 3.3. Run: 
		python randomMDPs_main.py #name_of_experiment# #random_seed#

We DO NOT include the following:
- The hyper-parameter search (Appendix C.2): it should be easy to re-implement.
- The figure generator: it has too many specificities to be made understandable for a user at the moment. Also, it is not hard to re-implement one's own visualization tools.
- The multi-CPU implementation: its structure is too much dependent on the cluster tools: it would be useless for somebody from another lab and might divulge author affiliations.


## License

This project is BSD-licensed.
