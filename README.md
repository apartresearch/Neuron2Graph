# Neuron to Graph: Interpreting Language Model Neurons at Scale

Authors: Alex Foote, Neel Nanda, Esben Kran, Ionnis Konstas, Shay Cohen and Fazl Barez

Short paper accepted at RTML workshop at ICLR 2023: https://arxiv.org/abs/2304.12918

Longer preprint available here: https://arxiv.org/abs/2305.19911

## Description

We provide a new interpretability tool for Language Models, Neuron to Graph (N2G). N2G builds an intepretable graph representation of any neuron in a Language Model, which can be visualised for human interpretation and used to simulate the behaviour of the target neuron by predicting activations on input text, allowing for a direct measure of the quality of the representation by comparing to the ground truth activations of the neuron. The resulting graphs are a searchable and programmatically comparable representation, facilitating greater automation of interpretability research. 

| ![Architecture](https://github.com/alexjfoote/Neuron2Graph/blob/main/img/architecture_v4.png?raw=true) | 
|:--:| 
| **Overall architecture of N2G.** Activations of the target neuron on the dataset examples are retrieved (neuron and activating tokens in red). Prompts are pruned and the importance of each token for neuron activation is measured (important tokens in blue). Pruned prompts are augmented by replacing important tokens with high-probability substitutes using DistilBERT. The augmented set of prompts are converted to a graph. The output graph is a real example which activates on the token "except" when preceded by any of the other tokens. |

Given dataset examples that maximally activate a target neuron within a Language Model, N2G extracts the minimal sub-string required for activation, computes the saliency of each token for neuron activation, and creates additional examples by replacing important tokens with likely substitutes using DistilBERT. The set of enriched examples with token saliencies is then converted to a trie representing the tokens on which a neuron activates, as well as the context required for activation on these tokens. The trie can be used to process text and output token-level activations, which can be compared to the ground-truth activations of the neuron for automatic evaluation. A simplified version of this trie can also be visualised for human interpretation - activating tokens are coloured red according to how strongly they activate the neuron, and context tokens are coloured blue according to their importance for neuron activation. Once a model has been processed and a neuron graph has been generated for every neuron in the model, these graphs can be searched to identify neurons with particular properties, such as activating on a particular token when it occurs with another context token.

## Examples

| ![In-context](https://github.com/alexjfoote/Neuron2Graph/blob/main/img/in_context_graph.png?raw=true) | 
|:--:| 
| Neuron graph for an in-context learning neuron that activates on repeated token sequences. Identified by searching the graph representations for neurons which frequently have a repeated token in their context tokens as well as their activating tokens. |


| ![similar](https://github.com/alexjfoote/Neuron2Graph/blob/main/img/similar_graph.png?raw=true) | 
|:--:| 
| A neuron graph that occurs for a neuron in Layer 1 and a neuron in Layer 4 of the model. The neurons have identical behaviour, and were recognised as a similar pair through an automated graph comparison process. |

| ![import](https://github.com/alexjfoote/Neuron2Graph/blob/main/img/import_figure.png?raw=true) | 
|:--:| 
| Neurons related to programming syntax, specifically import statements. Top - Neuron graph illustrating import syntax for the Go programming language. Bottom Left: Neuron graph showing fundamental elements of Python import syntax. Bottom Right: Neuron graph for a neuron that responds to the imports of widely-used Python packages. |

| ![superposition](https://github.com/alexjfoote/Neuron2Graph/blob/main/img/3_15_graph.png?raw=true) | 
|:--:| 
| A neuron graph exhibiting polysemanticity, with three disconnected subgraphs each responding to a phrase in a different language. |


## Citation

If you use N2G in your research, please cite one of our papers:


``` 
@inproceedings{foote2023neuron2graph,   
title={Neuron to Graph: Interpreting Language Model Neurons at Scale},   
author={Foote, Alex and Nanda, Neel and Kran, Esben and Konstas, Ionnis and Cohen, Shay and Barez, Fazl},   
booktitle={arXiv},   
year={2023} 
} 
```

``` 
@inproceedings{foote2023n2g,   
title={N$2$G: A scalable approach for quantifying interpretable neuron representations in Large Language Models},   
author={Foote, Alex and Nanda, Neel and Kran, Esben and Konstas, Ionnis and Barez, Fazl},   
booktitle={RTML workshop ICLR},   
year={2023} 
} 
```
