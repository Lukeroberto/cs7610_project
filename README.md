# CS7610 Project: Decentralized RL 

Proposal and paper for this project located in writeups.

This project is based loosely off the system described in the paper "Diff-DAC: Distributed Actor-Critic for Average Multitask Deep Reinforcement Learning": [Link](https://www.prowler.io/blog/diff-dac-fully-distributed-deep-reinforcement-learning).

In order to get started, run:

    pip install -r < requirements.txt

To run test cases, use the -m flag: 

Ex:

    python -m test.fail_stop_test -t "fail_stop" -l 2000

Where the "-t" flag refers to the test name (for plotting purposes) and the -l refers to the training length (number of episodes)