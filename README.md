# Transformer-Activation-Tools

Tools for exploring Transformer neuron behaviour, including input pruning and diversification, measuring token importance, and visualising the results.

See `Demo.ipynb` for a walkthrough of the method, and run it in Colab to try it yourself!

## Description

Given an dataset example that is highly activating to a given neuron, the algorithm will prune it to the shortest string that is still highly activating, then generate new variations of that pruned prompt by substituting tokens using BERT. It can measure token importance by masking tokens and measuring the change in activation, and visualise token importance and relative neuron activation for a prompt.

For example, given an input like text #0 for [this neuron](https://lexoscope.io/solu-8l-old/3/1.html) the algorithm produces and visualises amore diverse set of prompts, to give better insight into neuron behaviour.

![An example visualisation](Example.png)
