This repository contains my work for CS4100 Artificial Intelligence. It contains definitions for several reinforcement learning agents in the policy gradient based class of algorithms. The first stepping stone was the implementation of the REINFORCE algorithm, supported by the numpy library and extended to deeper networks as well as GPU compute with PyTorch. You can find the networks and inference code for the policies and state-value evaluation code in the two main modules 'policy' and 'value'. These are used to build the top level of algorithms in 'agents' in a pluggable fashion. 

Todo list:
- separate out gym code
- better abstractions for value functions
- interface for policies that has better interoperability with tensor types.
- more sophisticated batching
- layer norm
