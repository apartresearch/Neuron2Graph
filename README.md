# Transformer-Activation-Tools

Tools for exploring Transformer neuron behaviour, including input pruning and diversification, measuring token importance, and visualising the results.

See `Demo.ipynb` for a walkthrough of the method, and run it in Colab to try it yourself!

This project started as a submission to the [interpretability hackathon](https://itch.io/jam/interpretability), and ended up getting 1st place. See the [results](https://itch.io/jam/interpretability/results) and a [write-up](https://www.lesswrong.com/posts/hhhmcWkgLwPmBuhx7/results-from-the-interpretability-hackathon) for more info.

## Description

Given an dataset example that is highly activating to a given neuron, the algorithm will prune it to the shortest string that is still highly activating, then generate new variations of that pruned prompt by substituting tokens using BERT. It can measure token importance by masking tokens and measuring the change in activation, and visualise token importance and relative neuron activation for a prompt.

For example, given an input like text #0 for [this neuron](https://lexoscope.io/solu-8l-old/3/1.html) the algorithm produces and visualises a more diverse set of prompts, to give better insight into neuron behaviour.

![An example visualisation](img/Example.png)
