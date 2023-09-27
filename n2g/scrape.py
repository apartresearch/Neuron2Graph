import re
import requests
import json
import os
import multiprocessing as mp
import numpy as np

from functools import partial

act_parser = re.compile('<h4>Max Act: <b>')


def get_max_activations(model_name, layer, neuron, n=1):
    """Get the max activating dataset examples for a given neuron in a model"""
    base_url = f"https://neuroscope.io/{model_name}/{layer}/{neuron}.html"

    response = requests.get(base_url)
    webpage = response.text

    parts = act_parser.split(webpage)
    activations = []
    for i, part in enumerate(parts):
        if i == 0:
            continue

        activation = float(part.split('</b>')[0])

        activations.append(activation)
        if len(activations) >= n:
            break

    if len(activations) != min(20, n):
        raise Exception
    return activations if n > 1 else activations[0]


def get_max_acts(model_name, layer_and_neurons):
    layer, neurons = layer_and_neurons
    activations = []
    for i, neuron in enumerate(neurons):
        if i % 50 == 0:
            print(f"\nLayer {layer}: {i} of {len(neurons)} complete")
        try:
            activation = get_max_activations(model_name, layer, neuron, n=1)
            activations.append(activation)
        except:
            print(f"Neuron {neuron} in layer {layer} failed")
            # Use the previous activation as a hack to get around failures
            activations.append(activations[-1])
    return activations


if __name__ == "__main__":
    """
    Instructions:
    Look at https://neuroscope.io/ for the model you want to scrape
    Set the number of layers and neurons appropriately
    """
    base_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__))))

    model_name = "gpt2-small"
    layers = 24
    neurons = 4096

    info = [(layer, [neuron for neuron in range(neurons)]) for layer in range(layers)]

    with mp.Pool(layers) as p:
        activation_matrix = p.map(partial(get_max_acts, model_name), info)

    activation_matrix_np = np.array(activation_matrix)

    with open(os.path.join(base_path, f"data/activation_matrix-{model_name}.json"), "w") as ofh:
        json.dump(activation_matrix, ofh, indent=2, ensure_ascii=False)
