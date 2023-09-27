# Import stuff
import torch

from pathlib import Path

from transformer_lens import HookedTransformer, HookedTransformerConfig

import nltk

nltk.download('stopwords')

from transformers import AutoModelForCausalLM
from transformers import AutoModelForMaskedLM
from transformers import AutoTokenizer

import requests
import re

parser = re.compile('\{\"tokens\": ')


def get_snippets(model_name, layer, neuron):
    """Get the max activating dataset examples for a given neuron in a model"""
    base_url = f"https://neuroscope.io/{model_name}/{layer}/{neuron}.html"

    response = requests.get(base_url)
    webpage = response.text

    parts = parser.split(webpage)
    snippets = []
    for i, part in enumerate(parts):
        if i == 0 or i % 2 != 0:
            continue

        token_str = part.split(', "values": ')[0]

        tokens = json.loads(token_str)

        snippet = "".join(tokens)

        snippets.append(snippet)

    if len(snippets) != 20:
        raise Exception
    return snippets


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


class WordTokenizer:
    """Simple tokenizer for splitting text into words"""

    def __init__(self, split_tokens, stick_tokens):
        self.split_tokens = split_tokens
        self.stick_tokens = stick_tokens

    def __call__(self, text):
        return self.tokenize(text)

    def is_split(self, char):
        """Split on any non-alphabet chars unless excluded, and split on any specified chars"""
        return char in self.split_tokens or (not char.isalpha() and char not in self.stick_tokens)

    def tokenize(self, text):
        """Tokenize text, preserving all characters"""
        tokens = []
        current_token = ""
        for char in text:
            if self.is_split(char):
                tokens.append(current_token)
                tokens.append(char)
                current_token = ""
                continue
            current_token += char
        tokens.append(current_token)
        tokens = [token for token in tokens if token]
        return tokens


def layer_index_to_name(layer_index):
    return f"blocks.{layer_index}.{layer_ending}"


from pprint import pprint

splitter = re.compile("[\.!\\n]")


def sentence_tokenizer(str_tokens):
    """Split tokenized text into sentences"""
    sentences = []
    sentence = []
    sentence_to_token_indices = defaultdict(list)
    token_to_sentence_indices = {}

    for i, str_token in enumerate(str_tokens):
        sentence.append(str_token)
        sentence_to_token_indices[len(sentences)].append(i)
        token_to_sentence_indices[i] = len(sentences)
        if splitter.search(str_token) is not None or i + 1 == len(str_tokens):
            sentences.append(sentence)
            sentence = []

    return sentences, sentence_to_token_indices, token_to_sentence_indices


import math


def batch(arr, n=None, batch_size=None):
    if n is None and batch_size is None:
        raise ValueError("Either n or batch_size must be provided")
    if n is not None and batch_size is not None:
        raise ValueError("Either n or batch_size must be provided, not both")

    if n is not None:
        batch_size = math.floor(len(arr) / n)
    elif batch_size is not None:
        n = math.ceil(len(arr) / batch_size)

    extras = len(arr) - (batch_size * n)
    groups = []
    group = []
    added_extra = False
    for element in arr:
        group.append(element)
        if len(group) >= batch_size:
            if extras and not added_extra:
                extras -= 1
                added_extra = True
                continue
            groups.append(group)
            group = []
            added_extra = False

    if group:
        groups.append(group)

    return groups


from nltk.corpus import stopwords
from string import punctuation
from scipy.special import softmax


class FastAugmenter:
    """Uses BERT to generate variations on input text by masking words and substituting with most likely predictions"""

    def __init__(self, model, model_tokenizer, word_tokenizer, neuron_model, device="cuda:0"):
        self.model = model
        self.model_tokenizer = model_tokenizer
        self.stops = set(stopwords.words('english'))
        self.punctuation_set = set(punctuation)
        self.to_strip = " " + punctuation
        self.word_tokenizer = word_tokenizer
        self.device = device

    def augment(self, text, max_char_position=None, exclude_stopwords=False, n=5, important_tokens=None, **kwargs):
        joiner = ""
        tokens = self.word_tokenizer(text)

        new_texts = []
        positions = []

        important_tokens = {token.strip(self.to_strip).lower() for token in important_tokens}

        seen_prompts = set()

        # Gather all tokens to be substituted
        tokens_to_sub = []

        # Mask important tokens
        masked_token_sets = []
        masked_texts = []

        masked_tokens = []

        for i, token in enumerate(tokens):
            norm_token = token.strip(self.to_strip).lower() if any(c.isalpha() for c in token) else token

            if not token or word_tokenizer.is_split(token) or (exclude_stopwords and norm_token in self.stops) or (
                    important_tokens is not None and norm_token not in important_tokens):
                continue

            # If no alphanumeric characters, we'll do a special substitution rather than using BERT
            if not any(c.isalpha() for c in token):
                continue

            before = tokens[:i]
            before_text = joiner.join(before)
            position = len(before_text)

            # Don't bother if we're beyond the max activating token, as these tokens have no effect on the activation
            if max_char_position is not None and position > max_char_position:
                break

            copy_tokens = copy.deepcopy(tokens)
            copy_tokens[i] = "[MASK]"
            masked_token_sets.append((copy_tokens, position))
            masked_texts.append(joiner.join(copy_tokens))

            masked_tokens.append(token)

        # pprint(masked_texts)
        if len(masked_texts) == 0:
            return [], []

        inputs = self.model_tokenizer(masked_texts, padding=True, return_tensors="pt").to(self.device)
        token_probs = softmax(self.model(**inputs).logits.cpu().detach().numpy(), axis=-1)
        inputs = inputs.to("cpu")

        chosen_tokens = set()

        new_texts = []
        positions = []

        seen_texts = set()

        for i, (masked_token_set, char_position) in enumerate(masked_token_sets):
            mask_token_index = np.argwhere(inputs["input_ids"][i] == self.model_tokenizer.mask_token_id)[0, 0]

            mask_token_probs = token_probs[i, mask_token_index, :]

            # We negate the array before argsort to get the largest, not the smallest, logits
            top_probs = -np.sort(-mask_token_probs).transpose()
            top_tokens = np.argsort(-mask_token_probs).transpose()

            subbed = 0

            # Substitute the given token with the best predictions
            for l, (top_token, top_prob) in enumerate(zip(top_tokens, top_probs)):
                if top_prob < 0.00001:
                    break

                candidate_token = self.model_tokenizer.decode(top_token)

                # print(candidate_token)

                # Check that the predicted token isn't the same as the token that was already there
                normalised_candidate = candidate_token.strip(
                    self.to_strip).lower() if candidate_token not in self.punctuation_set else candidate_token
                normalised_token = token.strip(self.to_strip).lower() if token not in self.punctuation_set else token

                if normalised_candidate == normalised_token or not any(c.isalpha() for c in candidate_token):
                    continue

                # Get most common casing of the word
                most_common_casing = word_to_casings.get(candidate_token, [(candidate_token, 1)])[0][0]

                original_token = masked_tokens[i]
                # Title case normally has meaning (e.g., start of sentence, in a proper noun, etc.) so follow original token, otherwise use most common
                best_casing = candidate_token.title() if original_token.istitle() else most_common_casing

                new_token_set = copy.deepcopy(masked_token_set)
                # BERT uses ## to denote a tokenisation within a word, so we remove it to glue the word back together
                masked_text = joiner.join(new_token_set)
                new_text = masked_text.replace(self.model_tokenizer.mask_token, best_casing, 1).replace(" ##", "")

                if new_text in seen_texts:
                    continue

                new_texts.append(new_text)
                positions.append(char_position)
                subbed += 1

                if subbed >= n:
                    break

        return new_texts, positions


def augment(model, layer, index, prompt, aug, max_length=1024, inclusion_threshold=-0.5, exclusion_threshold=-0.5, n=5,
            **kwargs):
    """Generate variations of a prompt using an augmenter"""
    prepend_bos = True
    tokens = model.to_tokens(prompt, prepend_bos=prepend_bos)
    str_tokens = model.to_str_tokens(prompt, prepend_bos=prepend_bos)

    # print(prompt)

    if len(tokens[0]) > max_length:
        tokens = tokens[0, :max_length].unsqueeze(0)

    logits, cache = model.run_with_cache(tokens)
    activations = cache[layer][0, :, index]

    initial_max = torch.max(activations).cpu().item()
    initial_argmax = torch.argmax(activations).cpu().item()
    max_char_position = len("".join(str_tokens[int(prepend_bos):initial_argmax + 1]))

    positive_prompts = [(prompt, initial_max, 1)]
    negative_prompts = []

    if n == 0:
        return positive_prompts, negative_prompts

    aug_prompts, aug_positions = aug.augment(prompt, max_char_position=max_char_position, n=n, **kwargs)
    if not aug_prompts:
        return positive_prompts, negative_prompts

    aug_tokens = model.to_tokens(aug_prompts, prepend_bos=prepend_bos)

    aug_logits, aug_cache = model.run_with_cache(aug_tokens)
    all_aug_activations = aug_cache[layer][:, :, index]

    for aug_prompt, char_position, aug_activations in zip(aug_prompts, aug_positions, all_aug_activations):
        aug_max = torch.max(aug_activations).cpu().item()
        aug_argmax = torch.argmax(aug_activations).cpu().item()

        # TODO implement this properly - when we mask multiple tokens, if they cross the max_char_position this will not necessarily be correct
        if char_position < max_char_position:
            new_str_tokens = model.to_str_tokens(aug_prompt, prepend_bos=prepend_bos)
            aug_argmax += len(new_str_tokens) - len(str_tokens)

        proportion_drop = (aug_max - initial_max) / initial_max

        if proportion_drop >= inclusion_threshold:
            positive_prompts.append((aug_prompt, aug_max, proportion_drop))
        elif proportion_drop < exclusion_threshold:
            negative_prompts.append((aug_prompt, aug_max, proportion_drop))

    return positive_prompts, negative_prompts


def fast_prune(model, layer, neuron, prompt, max_length=1024, proportion_threshold=-0.5, absolute_threshold=None,
               token_activation_threshold=0.75, window=0, return_maxes=False, cutoff=30, batch_size=4,
               max_post_context_tokens=5, skip_threshold=0, skip_interval=5, return_intermediates=False, **kwargs):
    """Prune an input prompt to the shortest string that preserves x% of neuron activation on the most activating token."""

    prepend_bos = True
    tokens = model.to_tokens(prompt, prepend_bos=prepend_bos)
    str_tokens = model.to_str_tokens(prompt, prepend_bos=prepend_bos)

    if len(tokens[0]) > max_length:
        tokens = tokens[0, :max_length].unsqueeze(0)

    logits, cache = model.run_with_cache(tokens)
    activations = cache[layer][0, :, neuron].cpu()

    full_initial_max = torch.max(activations).cpu().item()
    full_initial_argmax = torch.argmax(activations).cpu().item()

    sentences, sentence_to_token_indices, token_to_sentence_indices = sentence_tokenizer(str_tokens)

    # print(activation_threshold * full_initial_max)

    strong_indices = torch.where(activations >= token_activation_threshold * full_initial_max)[0]
    strong_activations = activations[strong_indices].cpu()
    strong_indices = strong_indices.cpu()

    # print(strong_activations)
    # print(strong_indices)

    strong_sentence_indices = [token_to_sentence_indices[index.item()] for index in strong_indices]

    pruned_sentences = []
    final_max_indices = []
    all_intermediates = []
    initial_maxes = []
    truncated_maxes = []

    for strong_sentence_index, initial_argmax, initial_max in zip(strong_sentence_indices, strong_indices,
                                                                  strong_activations):
        initial_argmax = initial_argmax.item()
        initial_max = initial_max.item()
        # print(strong_sentence_index, initial_argmax, initial_max)

        max_sentence_index = token_to_sentence_indices[initial_argmax]
        relevant_str_tokens = [str_token for sentence in sentences[:max_sentence_index + 1] for str_token in sentence]

        prior_context = relevant_str_tokens[:initial_argmax + 1]

        post_context = relevant_str_tokens[initial_argmax + 1:]

        shortest_successful_prompt = None
        final_max_index = None

        truncated_prompts = []
        added_tokens = []

        count = 0
        full_prior = prior_context[:max(0, initial_argmax - window + 1)]

        for i, str_token in reversed(list(enumerate(full_prior))):
            count += 1

            if count > cutoff:
                break

            # print(count, len(full_prior))

            if not count == len(full_prior) and count >= skip_threshold and count % skip_interval != 0:
                continue

            # print("Made it!")

            truncated_prompt = prior_context[i:]
            joined = "".join(truncated_prompt)
            truncated_prompts.append(joined)
            added_tokens.append(i)

        batched_truncated_prompts = batch(truncated_prompts, batch_size=batch_size)
        batched_added_tokens = batch(added_tokens, batch_size=batch_size)

        finished = False
        intermediates = []
        for i, (truncated_batch, added_tokens_batch) in enumerate(zip(batched_truncated_prompts, batched_added_tokens)):
            # print("length", len(truncated_batch))
            # pprint(truncated_batch)

            truncated_tokens = model.to_tokens(truncated_batch, prepend_bos=prepend_bos)

            # pprint(truncated_tokens)

            logits, cache = model.run_with_cache(truncated_tokens)
            all_truncated_activations = cache[layer][:, :, neuron].cpu()

            # print("shape", all_truncated_activations.shape)

            for j, truncated_activations in enumerate(all_truncated_activations):
                num_added_tokens = added_tokens_batch[j]
                # print("single shape", truncated_activations.shape)
                truncated_argmax = torch.argmax(truncated_activations).cpu().item() + num_added_tokens
                final_max_index = torch.argmax(truncated_activations).cpu().item()

                if prepend_bos:
                    truncated_argmax -= 1
                    final_max_index -= 1
                truncated_max = torch.max(truncated_activations).cpu().item()

                # trunc_logits, trunc_cache = model.run_with_cache(model.to_tokens(truncated_batch[j], prepend_bos=prepend_bos))
                # trunc_activations = trunc_cache[layer][0, :, neuron]

                # print(truncated_activations)
                # print(trunc_activations)
                # print("truncated_argmax", truncated_argmax)
                # print(truncated_max)

                shortest_prompt = truncated_batch[j]

                if not shortest_prompt.startswith("<|endoftext|>"):
                    truncated_str_tokens = model.to_str_tokens(truncated_batch[j], prepend_bos=False)
                    intermediates.append((shortest_prompt, truncated_str_tokens[0], truncated_max))

                if (truncated_argmax == initial_argmax and (
                        (truncated_max - initial_max) / initial_max > proportion_threshold or
                        (absolute_threshold is not None and truncated_max >= absolute_threshold))) or (
                        i == len(batched_truncated_prompts) - 1 and j == len(all_truncated_activations) - 1):
                    shortest_successful_prompt = shortest_prompt
                    finished = True
                    break

            if finished:
                break

        # if shortest_successful_prompt is None:
        #   pruned_sentence = "".join(relevant_str_tokens)
        #   final_max_index = initial_argmax
        # else:
        pruned_sentence = "".join(
            shortest_successful_prompt)  # if shortest_successful_prompt is not None else shortest_prompt

        if max_post_context_tokens is not None:
            pruned_sentence += "".join(post_context[:max_post_context_tokens])

        pruned_sentences.append(pruned_sentence)
        final_max_indices.append(final_max_index)
        initial_maxes.append(initial_max)
        truncated_maxes.append(truncated_max)
        all_intermediates.append(intermediates)

    if return_maxes:
        return list(zip(pruned_sentences, final_max_indices, initial_maxes, truncated_maxes))

    elif return_intermediates:
        return list(zip(pruned_sentences, all_intermediates))

    return list(zip(pruned_sentences, final_max_indices))


import copy


def fast_measure_importance(model, layer, neuron, prompt, initial_argmax=None, max_length=1024, max_activation=None,
                            masking_token=1, threshold=0.8, scale_factor=1, return_all=False, activation_threshold=0.1,
                            **kwargs):
    """Compute a measure of token importance by masking each token and measuring the drop in activation on the max activating token"""

    prepend_bos = True
    tokens = model.to_tokens(prompt, prepend_bos=prepend_bos)
    str_tokens = model.to_str_tokens(prompt, prepend_bos=prepend_bos)

    if len(tokens[0]) > max_length:
        tokens = tokens[0, :max_length].unsqueeze(0)

    # logits, cache = model.run_with_cache(tokens)

    # print(tokens_and_activations)

    importances_matrix = []

    shortest_successful_prompt = None
    # cutoff = 50

    masked_prompts = tokens.repeat(len(tokens[0]) + 1, 1)

    # print(f"{len(masked_prompts)=}, {initial_argmax=}, {starting_point=}")

    for i in range(1, len(masked_prompts)):
        masked_prompts[i, i - 1] = masking_token

    # for i, str_token in enumerate(str_tokens):
    #   if i >= cutoff:
    #     break

    #   masked_tokens = tokens

    #   if i >= len(masked_tokens[0]):
    #     continue

    #   token_to_mask = copy.deepcopy(tokens[0, i])
    #   masked_tokens[0, i] = masking_token

    #   masked_prompts.append(masked_tokens[0])
    #   tokens[0, i] = token_to_mask

    # pprint(masked_prompts)

    logits, cache = model.run_with_cache(masked_prompts)
    all_masked_activations = cache[layer][1:, :, neuron].cpu()

    activations = cache[layer][0, :, neuron].cpu()

    if initial_argmax is None:
        initial_argmax = torch.argmax(activations).cpu().item()
    else:
        # This could be wrong
        initial_argmax = min(initial_argmax, len(activations) - 1)

    # print(activations)
    # print(activation_threshold)
    # activation_indexes = [i for i, activation in enumerate(activations) if activation * scale_factor / max_activation > activation_threshold]
    # print(activation_indexes)
    # final_activating = initial_argmax if len(activation_indexes) == 0 else activation_indexes[-1]

    initial_max = activations[initial_argmax].cpu().item()

    if max_activation is None:
        max_activation = initial_max
    scale = min(1, initial_max / max_activation)

    # print("scale_factor measure_importance", scale_factor)

    tokens_and_activations = [[str_token, round(activation.cpu().item() * scale_factor / max_activation, 3)] for
                              str_token, activation in zip(str_tokens, activations)]
    important_tokens = []
    tokens_and_importances = [[str_token, 0] for str_token in str_tokens]

    for i, masked_activations in enumerate(all_masked_activations):
        if return_all:
            # Get importance of the given token for all tokens
            importances_row = []
            for j, activation in enumerate(masked_activations):
                activation = activation.cpu().item()
                normalised_activation = (1 - (activation / activations[j].cpu().item()))
                importances_row.append((str_tokens[j], normalised_activation))

            # for j, str_token in enumerate(str_tokens[cutoff:]):
            #   importances_row.append((str_token, 0))

            # print("importances_row", importances_row)
            importances_matrix.append(np.array(importances_row))

        masked_max = masked_activations[initial_argmax].cpu().item()
        normalised_activation = (1 - (masked_max / initial_max))

        str_token = tokens_and_importances[i][0]
        tokens_and_importances[i][1] = normalised_activation
        if normalised_activation >= threshold and str_token != "<|endoftext|>":
            important_tokens.append(str_token)

    # for i, str_token in enumerate(str_tokens[cutoff:]):
    #   tokens_and_importances.append((str_token, 0))

    if return_all:
        # Flip so we have the importance of all tokens for a given token
        importances_matrix = np.array(importances_matrix)
        return importances_matrix, initial_max, important_tokens, tokens_and_activations, initial_argmax

    return tokens_and_importances, initial_max, important_tokens, tokens_and_activations, initial_argmax


import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


def visualise(tokens_and_activations, tokens_and_importances, max_index=None, title=None, truncate=False,
              labels=["Activation", "Importance"], **kwargs):
    """Visualise relative token activation and importance"""
    if max_index is None:
        max_index = len(tokens_and_activations)

    zero_width = u'\u200b'
    token_counter = Counter()
    data = {}
    count = 0

    for i, ((token, importance), (_, activation)) in enumerate(zip(tokens_and_importances, tokens_and_activations)):
        if token == "<|endoftext|>":
            continue

        if i > max_index and truncate:
            break

        # This is a horrible hack to allow us to have a dict with the "same" token as multiple keys - by adding zero width spaces the tokens look the same but are actually different
        seen_count = token_counter[token]
        add = zero_width * seen_count
        deduped_token = token + add
        # Have to escape dollars so matplotlib doesn't interpret them as latex
        deduped_token = deduped_token.replace("$", "\$")
        data[deduped_token] = [activation, importance]
        token_counter[token] += 1
        count += 1

    df = pd.DataFrame(data, index=labels)
    plt.figure(figsize=[int(count * 1.5), 1.2])
    sns.heatmap(df, vmin=0, vmax=1, xticklabels=True, annot=True)

    if title is not None:
        title = title.replace("$", "\$")
        plt.title(title)


from sklearn.model_selection import train_test_split


def train_and_eval(model, layer, neuron, aug, train_proportion=0.5, max_train_size=10, max_eval_size=20,
                   fire_threshold=0.5, random_state=0, train_indexes=None, return_paths=False, **kwargs):
    if isinstance(layer, int):
        layer = layer_index_to_name(layer)

    layer_num = int(layer.split(".")[1])
    base_max_act = float(activation_matrix[layer_num, neuron])

    snippets = get_snippets(model_name, layer_num, neuron)

    if train_indexes is None:
        train_snippets, test_snippets = train_test_split(snippets, train_size=train_proportion,
                                                         random_state=random_state)
    else:
        train_snippets = [snippet for i, snippet in enumerate(snippets) if i in train_indexes]
        test_snippets = [snippet for i, snippet in enumerate(snippets) if i not in train_indexes]
    # train_data, test_data = train_test_split(data, train_size=train_proportion, random_state=0)

    # train_data_snippets = ["".join(tokens) for tokens, activations in train_data if any(activation > fire_threshold for activation in activations)][:max_train_size]
    train_data_snippets = []
    all_train_snippets = train_snippets + train_data_snippets

    all_info = []
    for i, snippet in enumerate(all_train_snippets):
        # if i % 10 == 0:
        print(f"Processing {i + 1} of {len(all_train_snippets)}")

        pruned_results = fast_prune(model, layer, neuron, snippet, return_maxes=True, **kwargs)

        for pruned_prompt, _, initial_max_act, truncated_max_act in pruned_results:
            # tokens = model.to_tokens(pruned_prompt, prepend_bos=True)
            # str_tokens = model.to_str_tokens(pruned_prompt, prepend_bos=True)
            # logits, cache = model.run_with_cache(tokens)
            # activations = cache[layer][0, :, neuron].cpu()
            # max_pruned_activation = torch.max(activations).item()
            scale_factor = initial_max_act / truncated_max_act
            # scale_factor = 1

            # print(scale_factor)
            # scaled_activations = activations * scale_factor / base_max_act

            # print(list(zip(str_tokens, activations)))

            # print(pruned_prompt)

            # print(len(pruned_prompt))

            if pruned_prompt is None:
                continue

            info = augment_and_return(model, layer, neuron, aug, pruned_prompt, base_max_act=base_max_act,
                                      scale_factor=scale_factor, **kwargs)
            all_info.append(info)

    neuron_model = NeuronModel(layer_num, neuron, **kwargs)
    paths = neuron_model.fit(all_info)

    print("Fitted model")

    max_test_data = []
    for snippet in test_snippets:
        # pruned_prompt, _ = prune(model, layer, neuron, snippet, **kwargs)
        # if pruned_prompt is None:
        #   continue
        tokens = model.to_tokens(snippet, prepend_bos=True)
        str_tokens = model.to_str_tokens(snippet, prepend_bos=True)
        logits, cache = model.run_with_cache(tokens)
        activations = cache[layer][0, :, neuron].cpu()
        max_test_data.append((str_tokens, activations / base_max_act))

    # pprint(max_test_data[0])
    # print("\n\n")
    # pprint(test_data[0])

    # print("Evaluation data")
    # test_data = test_data[:max_eval_size]
    # evaluate(neuron_model, test_data, fire_threshold=fire_threshold, **kwargs)

    print("Max Activating Evaluation Data")
    try:
        stats = evaluate(neuron_model, max_test_data, fire_threshold=fire_threshold, **kwargs)
    except Exception as e:
        stats = {}
        print(f"Stats failed with error: {e}")

    if return_paths:
        return stats, paths
    return stats


def augment_and_return(model, layer, neuron, aug, pruned_prompt, base_max_act=None, use_index=False, scale_factor=1,
                       **kwargs):
    info = []
    importances_matrix, initial_max_act, important_tokens, tokens_and_activations, initial_max_index = fast_measure_importance(
        model, layer, neuron, pruned_prompt, max_activation=base_max_act, scale_factor=scale_factor, return_all=True)

    if base_max_act is not None:
        initial_max_act = base_max_act

    positive_prompts, negative_prompts = augment(model, layer, neuron, pruned_prompt, aug,
                                                 important_tokens=set(important_tokens), **kwargs)

    for i, (prompt, activation, change) in enumerate(positive_prompts):
        title = prompt
        if i == 0:
            title = "Original - " + prompt

        #   print("Original")
        #   print(prompt, "\n")
        # elif i > 1:
        #   print("Augmented")
        #   print(prompt, "\n")

        if use_index:
            importances_matrix, max_act, _, tokens_and_activations, max_index = fast_measure_importance(model, layer,
                                                                                                        neuron, prompt,
                                                                                                        max_activation=initial_max_act,
                                                                                                        initial_argmax=initial_max_index,
                                                                                                        scale_factor=scale_factor,
                                                                                                        return_all=True)
        else:
            importances_matrix, max_act, _, tokens_and_activations, max_index = fast_measure_importance(model, layer,
                                                                                                        neuron, prompt,
                                                                                                        max_activation=initial_max_act,
                                                                                                        scale_factor=scale_factor,
                                                                                                        return_all=True)
        info.append((importances_matrix, tokens_and_activations, max_index))

    for prompt, activation, change in negative_prompts:
        if use_index:
            importances_matrix, max_act, _, tokens_and_activations, max_index = fast_measure_importance(model, layer,
                                                                                                        neuron, prompt,
                                                                                                        max_activation=initial_max_act,
                                                                                                        initial_argmax=initial_max_index,
                                                                                                        scale_factor=scale_factor,
                                                                                                        return_all=True)
        else:
            importances_matrix, max_act, _, tokens_and_activations, max_index = fast_measure_importance(model, layer,
                                                                                                        neuron, prompt,
                                                                                                        max_activation=initial_max_act,
                                                                                                        scale_factor=scale_factor,
                                                                                                        return_all=True)
        info.append((importances_matrix, tokens_and_activations, max_index))

    return info


def fast_augment_and_visualise(model, layer, neuron, aug, pruned_prompt, use_index=False, **kwargs):
    tokens_and_importances, max_act, important_tokens, tokens_and_activations, initial_max_index = fast_measure_importance(
        model, layer, neuron, pruned_prompt)

    positive_prompts, negative_prompts = augment(model, layer, neuron, pruned_prompt, aug,
                                                 important_tokens=set(important_tokens), **kwargs)
    for i, (prompt, activation, change) in enumerate(positive_prompts):
        title = prompt
        if i == 0:
            title = "Original - " + prompt
        if use_index:
            tokens_and_importances, _, _, tokens_and_activations, max_index = fast_measure_importance(model, layer,
                                                                                                      neuron, prompt,
                                                                                                      max_activation=max_act,
                                                                                                      initial_argmax=initial_max_index)
        else:
            tokens_and_importances, _, _, tokens_and_activations, max_index = fast_measure_importance(model, layer,
                                                                                                      neuron, prompt,
                                                                                                      max_activation=max_act)
        # visualise(tokens_and_activations, tokens_and_importances, max_index, title=title, **kwargs)

    for prompt, activation, change in negative_prompts:
        if use_index:
            tokens_and_importances, _, _, tokens_and_activations, max_index = fast_measure_importance(model, layer,
                                                                                                      neuron, prompt,
                                                                                                      max_activation=max_act,
                                                                                                      initial_argmax=initial_max_index)
        else:
            tokens_and_importances, _, _, tokens_and_activations, max_index = fast_measure_importance(model, layer,
                                                                                                      neuron, prompt,
                                                                                                      max_activation=max_act)
        # visualise(tokens_and_activations, tokens_and_importances, max_index, title=prompt, **kwargs)


def fast_run(model, layer, neuron, aug, snippets=None, num_examples=5, example_indexes=None, **kwargs):
    """For a given neuron, grab the max activating dataset examples, run them through the pruning and augmentation steps, and visualise the results"""
    if snippets is None:
        snippets = get_snippets(model_name, layer, neuron)
        if example_indexes is not None:
            snippets = [snippet for i, snippet in enumerate(snippets) if i in example_indexes]
        else:
            snippets = snippets[:num_examples]

    if isinstance(layer, int):
        layer = f"blocks.{layer}.{layer_ending}"

    for snippet in snippets:
        pruned_prompt, _ = fast_prune(model, layer, neuron, snippet, include_post_context=False, **kwargs)

        if pruned_prompt is None:
            continue

        fast_augment_and_visualise(model, layer, neuron, aug, pruned_prompt, **kwargs)


def layer_and_neuron_to_index(layer, neuron, width=3072, block_size=None):
    index = (layer * width) + neuron
    if block_size is None:
        return index
    return divmod(index, block_size)


def index_to_layer_and_neuron(index, width=3072):
    return divmod(index, width)


from sklearn.metrics import classification_report


def evaluate(neuron_model, data, fire_threshold=0.5, **kwargs):
    y = []
    y_pred = []
    y_act = []
    y_pred_act = []
    for prompt_tokens, activations in data:
        # print("truth")
        non_zero_indices = [i for i, activation in enumerate(activations) if activation > 0]
        start = max(0, non_zero_indices[0] - 10)
        end = min(len(prompt_tokens) - 1, non_zero_indices[-1] + 10)
        pred_activations = neuron_model.forward([prompt_tokens], return_activations=True)[0]

        y_act.extend(activations)
        y_pred_act.extend(pred_activations)

        important_context = list(zip(prompt_tokens, activations, pred_activations))[start:end]

        # print(important_context)
        # print(len(pred_activations))
        pred_firings = [int(pred_activation >= fire_threshold) for pred_activation in pred_activations]
        firings = [int(activation >= fire_threshold) for activation in activations]
        y_pred.extend(pred_firings)
        y.extend(firings)
    # print(len(y), len(y_pred))
    print(classification_report(y, y_pred))
    report = classification_report(y, y_pred, output_dict=True)

    y_act = np.array(y_act)
    y_pred_act = np.array(y_pred_act)

    # y_pred_act = y_pred_act[y_act > 0.5]
    # y_act = y_act[y_act > 0.5]

    # print(y_act[:10])
    # print(y_pred_act[:10])

    # y_pred_act = y_pred_act * np.mean(y_act) / np.mean(y_pred_act)
    # y_pred_act =

    act_diff = y_pred_act - y_act
    mse = np.mean(np.power(act_diff, 2))
    variance = np.var(y_act)
    correlation = 1 - (mse / variance)
    # print(f"{correlation=:.3f}, {mse=:.3f}, {variance=:.4f}")

    report["correlation"] = correlation
    return report


def train_and_eval_baseline(model, layer, neuron, Baseline, train_proportion=0.5, fire_threshold=0.5, random_state=0,
                            train_indexes=None, **kwargs):
    if isinstance(layer, int):
        layer = layer_index_to_name(layer)

    layer_num = int(layer.split(".")[1])

    base_max_act = float(activation_matrix[layer_num, neuron])

    snippets = get_snippets(model_name, layer_num, neuron)
    # data = get_data(layer_num, neuron)

    if train_indexes is None:
        train_snippets, test_snippets = train_test_split(snippets, train_size=train_proportion,
                                                         random_state=random_state)
    else:
        train_snippets = [snippet for i, snippet in enumerate(snippets) if i in train_indexes]
        test_snippets = [snippet for i, snippet in enumerate(snippets) if i not in train_indexes]
    # train_data, test_data = train_test_split(data, train_size=train_proportion, random_state=0)

    # train_data_snippets = ["".join(tokens) for tokens, activations in train_data if any(activation > fire_threshold for activation in activations)][:max_train_size]
    train_data_snippets = []
    all_train_snippets = train_snippets + train_data_snippets

    baseline_model = Baseline(model, layer_num, neuron, **kwargs)
    baseline_model.fit(all_train_snippets)

    print("Fitted model")

    # Not pruning so don't need to prepend_bos
    prepend_bos = False

    max_test_data = []
    for snippet in test_snippets:
        tokens = model.to_tokens(snippet, prepend_bos=prepend_bos)
        str_tokens = model.to_str_tokens(snippet, prepend_bos=prepend_bos)
        logits, cache = model.run_with_cache(tokens)
        activations = cache[layer][0, :, neuron]
        max_test_data.append((str_tokens, activations.cpu() / base_max_act))

    print("Max Activating Evaluation Data")
    # try:
    stats = evaluate(baseline_model, max_test_data, fire_threshold=fire_threshold, **kwargs)

    # except Exception as e:
    #   stats = {}
    #   print(f"Stats failed with error: {e}")

    return stats


def evaluate_baseline(baseline, folder_name, layers=6, neurons=3072, layer_start=0, neuron_start=0, **kwargs):
    random.seed(0)

    all_neuron_indices = [i for i in range(neurons)]

    all_stats = {}
    folder_path = os.path.join(base_path, f"neuron_graphs/{model_name}/{folder_name}")

    if not os.path.exists(folder_path):
        print("Making", folder_path)
        os.mkdir(folder_path)

    if os.path.exists(f"{folder_path}/stats.json"):
        with open(f"{folder_path}/stats.json") as ifh:
            all_stats = json.load(ifh)

    else:
        all_stats = {}

    for i, layer in enumerate(range(layer_start, layers)):
        if layer not in all_stats:
            all_stats[layer] = {}

        for j, neuron in enumerate(range(neuron_start, neurons)):
            print(f"{layer=} {neuron=}")
            try:
                stats = train_and_eval_baseline(model, layer, neuron, baseline, train_proportion=0.5,
                                                fire_threshold=0.5, **kwargs)

                all_stats[layer][neuron] = stats

                if j % 10 == 0:
                    with open(f"{folder_path}/stats.json", "w") as ofh:
                        json.dump(all_stats, ofh, indent=2)

            except Exception as e:
                print(e)
                print("Failed")

    with open(f"{folder_path}/stats.json", "w") as ofh:
        json.dump(all_stats, ofh, indent=2)


def get_summary_stats(path, verbose=True):
    summary_stats = []
    summary_stds = []

    with open(path) as ifh:
        stats = json.load(ifh)

    missing = 0

    random.seed(0)

    inelegible_count = 0

    precision_case = 0

    for layer, layer_stats in stats.items():
        # pprint(layer_stats)
        eligible_neurons = [neuron for neuron, neuron_stats in layer_stats.items() if "1" in neuron_stats]
        # neuron_sample = set(random.sample(eligible_neurons, 50))
        eligible_neurons = set(eligible_neurons)

        aggr_stats_dict = {"Inactivating": defaultdict(list), "Activating": defaultdict(list)}
        for neuron, neuron_stats in layer_stats.items():
            if neuron not in eligible_neurons:
                inelegible_count += 1
                continue

            aggr_stats_dict["Inactivating"]["Precision"].append(neuron_stats["0"]["precision"])
            aggr_stats_dict["Inactivating"]["Recall"].append(neuron_stats["0"]["recall"])
            aggr_stats_dict["Inactivating"]["F1"].append(neuron_stats["0"]["f1-score"])

            # print(neuron_stats["0"]["precision"], neuron_stats["0"]["recall"], neuron_stats["0"]["f1-score"],
            #       neuron_stats["1"]["precision"], neuron_stats["1"]["recall"], neuron_stats["1"]["f1-score"])

            # If we didn't predict anything as activating, treat this as 100% precision rather than 0%
            if neuron_stats["0"]["recall"] == 1 and neuron_stats["1"]["recall"] == 0:
                # print("Precision case")
                precision_case += 1
                neuron_stats["1"]["precision"] = 1.0

            aggr_stats_dict["Activating"]["Precision"].append(neuron_stats["1"]["precision"])
            aggr_stats_dict["Activating"]["Recall"].append(neuron_stats["1"]["recall"])
            aggr_stats_dict["Activating"]["F1"].append(neuron_stats["1"]["f1-score"])

        #   if neuron == "20":
        #     break
        # break

        # if neuron_stats["1"]["recall"] > 0.8:
        #   print(f'{layer}, {neuron}, {neuron_stats["1"]["precision"]:.3f}, {neuron_stats["1"]["recall"]:.3f}, {neuron_stats["1"]["f1-score"]:.3f}')
        if verbose:
            print("Neurons Evaluated:", len(aggr_stats_dict["Inactivating"]["Precision"]))

        avg_stats_dict = {"Inactivating": {}, "Activating": {}}
        std_stats_dict = {"Inactivating": {}, "Activating": {}}
        for token_type, inner_stats_dict in aggr_stats_dict.items():
            for stat_type, stat_arr in inner_stats_dict.items():
                avg_stats_dict[token_type][stat_type] = round(np.mean(stat_arr), 3)
                std_stats_dict[token_type][stat_type] = round(np.std(stat_arr), 3)

        summary_stats.append(avg_stats_dict)
        summary_stds.append(std_stats_dict)
        # break

    if verbose:
        for layer, (summary, std_summary) in enumerate(zip(summary_stats, summary_stds)):
            print("\n")
            pprint(summary)
            pprint(std_summary)

        print(f"{inelegible_count=}")
        print(f"{precision_case=}")

    return summary_stats


from collections import defaultdict, namedtuple, Counter
import json
from graphviz import Digraph, escape
from typing import List
import os
from IPython.display import Image, display


class NeuronStore:
    def __init__(self, path):
        if not os.path.exists(path):
            neuron_store = {
                "activating": {},
                "important": {}
            }
            with open(path, "w") as ofh:
                json.dump(neuron_store, ofh, indent=2, ensure_ascii=False)

        with open(path) as ifh:
            self.store = json.load(ifh)

        self.to_sets()
        self.path = path
        self.count_tokens()
        self.by_neuron()

    def save(self):
        self.to_lists()
        with open(self.path, "w") as ofh:
            json.dump(self.store, ofh, indent=2, ensure_ascii=False)
        self.to_sets()

    def to_sets(self):
        self.store = {token_type: {token: set(info) for token, info in token_dict.items()} for token_type, token_dict in
                      self.store.items()}

    def to_lists(self):
        self.store = {token_type: {token: list(set(info)) for token, info in token_dict.items()} for
                      token_type, token_dict in self.store.items()}

    def by_neuron(self):
        self.neuron_to_tokens = {}
        for token_type, token_dict in self.store.items():
            for token, neurons in token_dict.items():
                for neuron in neurons:
                    if neuron not in self.neuron_to_tokens:
                        self.neuron_to_tokens[neuron] = {"activating": set(), "important": set()}
                    self.neuron_to_tokens[neuron][token_type].add(token)

    def search(self, tokens_and_types):
        match_arr = []

        for token, token_type in tokens_and_types:
            token_types = [token_type] if token_type is not None else ["activating", "important"]
            token_matches = set()

            for token_type in token_types:
                matches = self.store[token_type].get(token, set())
                token_matches |= matches

            match_arr.append(token_matches)

        valid_matches = set.intersection(*match_arr)
        return valid_matches

    def count_tokens(self):
        self.neuron_individual_token_counts = defaultdict(Counter)
        self.neuron_total_token_counts = Counter()
        for token_type, token_dict in self.store.items():
            for token, neurons in token_dict.items():
                for neuron in neurons:
                    self.neuron_individual_token_counts[neuron][token] += 1
                    self.neuron_total_token_counts[neuron] += 1

    def find_similar(self, target_token_types=None, threshold=0.9):
        if target_token_types is None:
            target_token_types = {"activating", "important"}

        similar_pairs = []
        subset_pairs = []

        for i, (neuron_1, neuron_dict_1) in enumerate(self.neuron_to_tokens.items()):
            if i % 1000 == 0:
                print(f"{i} of {len(self.neuron_to_tokens.items())} complete")

            for j, (neuron_2, neuron_dict_2) in enumerate(self.neuron_to_tokens.items()):
                if i <= j:
                    continue

                all_similar = []
                all_subset = []

                for token_type in target_token_types:
                    length_1 = len(neuron_dict_1[token_type])
                    length_2 = len(neuron_dict_2[token_type])

                    intersection = neuron_dict_1[token_type] & neuron_dict_2[token_type]
                    similar = (len(intersection) / max(length_1, length_2, 1)) >= threshold
                    subset = len(intersection) / max(min(length_1, length_2), 1) >= threshold

                    all_similar.append(similar)
                    all_subset.append(subset)

                if all(all_similar):
                    similar_pairs.append((neuron_1, neuron_2))
                elif all(all_subset):
                    # The first token indicates the superset neuron and the second the subset neuron
                    subset_pair = (neuron_1, neuron_2) if length_2 < length_1 else (neuron_2, neuron_1)
                    subset_pairs.append(subset_pair)

        return similar_pairs, subset_pairs


def view_neuron(path):
    display(Image(filename=path))


class NeuronNode:
    def __init__(self, id_=None, value=None, children=None, depth=None, important=False, activator=False):
        if value is None:
            value = {}
        if children is None:
            children = {}
        self.id_ = id_
        self.value = value
        self.children = children
        self.depth = depth

    def __repr__(self):
        return f"ID: {self.id_}, Value: {json.dumps(self.value)}"

    def paths(self):
        if not self.children:
            return [[self.value]]  # one path: only contains self.value
        paths = []
        for child_token, child_tuple in self.children.items():
            child_node, _ = child_tuple
            for path in child_node.paths():
                paths.append([self.value] + path)
        return paths


class NeuronEdge:
    def __init__(self, weight=0, parent=None, child=None):
        self.weight = weight
        self.parent = parent
        self.child = child

    def __repr__(self):
        parent_str = json.dumps(self.parent.id_) if self.parent is not None else "None"
        child_str = json.dumps(self.child.id_) if self.child is not None else "None"
        return f"Weight: {self.weight:.3f}\nParent: {parent_str}\nChild: {child_str}"


class NeuronModel:
    def __init__(self, layer, neuron, activation_threshold=0.1, importance_threshold=0.5, folder_name=None,
                 neuron_store=None, **kwargs):
        self.layer = layer
        self.neuron = neuron
        self.Element = namedtuple("Element",
                                  "importance, activation, token, important, activator, ignore, is_end, token_value")
        self.neuron_store = neuron_store

        self.root_token = "**ROOT**"
        self.ignore_token = "**IGNORE**"
        self.end_token = "**END**"
        self.special_tokens = {self.root_token, self.ignore_token, self.end_token}

        self.root = (
            NeuronNode(-1, self.Element(0, 0, self.root_token, False, False, True, False, self.root_token), depth=-1),
            NeuronEdge())
        self.trie_root = (
            NeuronNode(-1, self.Element(0, 0, self.root_token, False, False, True, False, self.root_token), depth=-1),
            NeuronEdge())
        self.activation_threshold = activation_threshold
        self.importance_threshold = importance_threshold
        # self.net = Network(notebook=True)
        # self.net = Graph(graph_attr={"rankdir": "LR", "splines": "spline", "ranksep": "20", "nodesep": "1"}, node_attr={"fixedsize": "true", "width": "1.5"})
        # self.net = Graph(
        #     graph_attr={"rankdir": "RL", "splines": "spline", "ranksep": "5", "nodesep": "1"},
        #     node_attr={"fixedsize": "true", "width": "2"}
        # )
        # self.net = Graph(
        #     graph_attr={"rankdir": "RL", "splines": "spline", "ranksep": "2", "nodesep": "0.25"},
        #     node_attr={"fixedsize": "true", "width": "2", "height": "0.75"}
        # )
        self.net = Digraph(
            graph_attr={"rankdir": "RL", "splines": "spline", "ranksep": "1.5", "nodesep": "0.2"},
            node_attr={"fixedsize": "true", "width": "2", "height": "0.75"}
        )
        self.node_count = 0
        self.trie_node_count = 0
        self.max_depth = 0
        self.folder_name = folder_name

    def __call__(self, tokens_arr: List[List[str]]) -> List[List[float]]:
        return self.forward(tokens_arr)

    def fit(self, data):
        for example_data in data:
            for j, info in enumerate(example_data):
                if j == 0:
                    lines, important_index_sets = self.make_line(info)
                else:
                    lines, _ = self.make_line(info, important_index_sets)

                for line in lines:
                    # print("\nline", line)
                    self.add(self.root, line, graph=True)
                    self.add(self.trie_root, line, graph=False)

        # print("Paths before merge")
        # for path in self.trie_root[0].paths():
        #   print(path)

        self.build(self.root)
        self.merge_ignores()

        self.save_neurons()

        print("Paths after merge")
        paths = []
        for path in self.trie_root[0].paths():
            # print(path)
            paths.append(path)

        return paths

    def save_neurons(self):
        visited = set()  # List to keep track of visited nodes.
        queue = []  # Initialize a queue

        visited.add(self.trie_root[0].id_)
        queue.append(self.trie_root)

        while queue:
            node, edge = queue.pop(0)

            token = node.value.token

            if token not in self.special_tokens:
                add_dict = self.neuron_store.store["activating"] if node.value.activator else self.neuron_store.store[
                    "important"]
                if token not in add_dict:
                    add_dict[token] = set()
                add_dict[token].add(f"{self.layer}_{self.neuron}")

            for token, neighbour in node.children.items():
                new_node, new_edge = neighbour
                if new_node.id_ not in visited:
                    visited.add(new_node.id_)
                    queue.append(neighbour)

    @staticmethod
    def normalise(token):
        normalised_token = token.lower() if token.istitle() and len(token) > 1 else token
        normalised_token = normalised_token.strip() if len(normalised_token) > 1 and any(
            c.isalpha() for c in normalised_token) else normalised_token
        return normalised_token

    def make_line(self, info, important_index_sets=None):
        if important_index_sets is None:
            important_index_sets = []
            create_indices = True
        else:
            create_indices = False

        importances_matrix, tokens_and_activations, max_index = info

        # print(tokens_and_activations)

        all_lines = []

        for i, (token, activation) in enumerate(tokens_and_activations):
            if create_indices:
                important_index_sets.append(set())

            # if activation > 0.2:
            #   print([token], activation)

            if not activation > self.activation_threshold:
                continue

            # print("\ntoken", token)

            before = tokens_and_activations[:i + 1]

            line = []
            last_important = 0

            if not create_indices:
                # The if else is a bit of a hack to account for augmentations that have a different number of tokens to the original prompt
                important_indices = important_index_sets[i] if i < len(important_index_sets) else important_index_sets[
                    -1]
            else:
                important_indices = set()

            # print("before", before)

            for j, (seq_token, seq_activation) in enumerate(reversed(before)):
                if seq_token == "<|endoftext|>":
                    continue

                seq_index = len(before) - j - 1
                # Stop when we reach the last matrix entry, which corresponds to the last activating token
                # if seq_index >= len(importances_matrix):
                #   break
                important_token, importance = importances_matrix[seq_index, i]
                importance = float(importance)
                # print("importance", importance)

                important = importance > self.importance_threshold or (
                        not create_indices and seq_index in important_indices)
                activator = seq_activation > self.activation_threshold

                # print("important_index_sets[i]", important_index_sets[i])
                # print("create_indices", create_indices)
                # print("important", important)
                # print("seq_token", seq_token)
                # print("seq_index", seq_index)
                # print("important_token", important_token)

                if important and create_indices:
                    important_indices.add(seq_index)
                    # print("important_indices", important_indices)

                ignore = not important and j != 0
                is_end = False

                seq_token_identifier = self.ignore_token if ignore else seq_token

                new_element = self.Element(importance, seq_activation, seq_token_identifier, important, activator,
                                           ignore, is_end, seq_token)

                # print("new_element", new_element)

                if not ignore:
                    last_important = j

                line.append(new_element)

            line = line[:last_important + 1]
            # Add an end node
            line.append(self.Element(0, activation, self.end_token, False, False, True, True, self.end_token))
            # print(line)
            all_lines.append(line)

            if create_indices:
                important_index_sets[i] = important_indices

        # print("From", tokens_and_activations)
        # for line in all_lines:
        #   print("\nMade", line)

        return all_lines, important_index_sets

    def add(self, start_tuple, line, graph=True):
        current_tuple = start_tuple
        previous_element = None
        important_count = 0

        # print("starting at", current_tuple)
        # print("adding", line)

        start_depth = current_tuple[0].depth

        for i, element in enumerate(line):
            # print("\nelement", element)
            if element is None and i > 0:
                break

            # importance, activation, token, important, activator, ignore, is_end = element

            if element.ignore and graph:
                continue

            # Normalise token
            element = element._replace(token=self.normalise(element.token))

            if graph:
                # Set end value as we don't have end nodes in the graph
                # The current node is an end if there's only one more node, as that will be the end node that we don't add
                is_end = i == len(line) - 2
                element = element._replace(is_end=is_end)

            important_count += 1

            current_node, current_edge = current_tuple

            if not current_node.value.ignore:
                prev_important_node = current_node

            # print("current_node", current_node)
            # print("children", current_node.children)

            if element.token in current_node.children:
                current_tuple = current_node.children[element.token]
                # print("Already in children")
                continue

            # if i == 0:
            #   weight = 0
            # # elif i == 1:
            #   # weight = previous_element.value["activation"] * element.value["importance"]
            # else:
            #   weight = prev_important_node.value.importance * element.importance
            weight = 0

            depth = start_depth + important_count
            new_node = NeuronNode(self.node_count, element, {}, depth=depth)
            new_tuple = (new_node, NeuronEdge(weight, current_node, new_node))

            self.max_depth = depth if depth > self.max_depth else self.max_depth
            # print(current_node)
            # print(new_node)

            current_node.children[element.token] = new_tuple

            # print("Added new node")
            # print("children", current_node.children)

            current_tuple = new_tuple

            self.node_count += 1

        return current_tuple

    # def merge(self, parent_tuple, merge_tuple):
    #   visited = set() # List to keep track of visited nodes.
    #   queue = []      # Initialize a queue

    #   visited.add(merge_tuple[0].id_)
    #   queue.append(merge_tuple)

    #   while queue:
    #     node, edge = queue.pop(0)

    #     parent_node, _ = parent_tuple

    #     parent_tuple = self.add(parent_node, [node.value])

    #     for token, neighbour in node.children.items():
    #       new_node, new_edge = neighbour
    #       if new_node.id_ not in visited:
    #         visited.add(new_node.id_)
    #         queue.append(neighbour)

    def merge_ignores(self):
        """
    Where a set of children contain an ignore token, merge the other nodes into it:
      - Fully merge if the other node is not an end node
      - Give the ignore node the other node's children (if it has any) if the other node is an end node
    """
        # print("\n\n******MERGING*******")
        visited = set()  # List to keep track of visited nodes.
        queue = []  # Initialize a queue

        visited.add(self.trie_root[0].id_)
        queue.append(self.trie_root)

        while queue:
            node, edge = queue.pop(0)

            token = node.value.token

            # print(node)

            if self.ignore_token in node.children:
                ignore_tuple = node.children[self.ignore_token]

                # print("ignore_tuple", ignore_tuple)

                to_remove = []

                for child_token, child_tuple in node.children.items():
                    if child_token == self.ignore_token:
                        continue

                    child_node, child_edge = child_tuple

                    child_paths = child_node.paths()

                    for path in child_paths:
                        # print("path", path)
                        # Don't merge if the path is only the first tuple, or the first tuple and an end tuple
                        if len(path) <= 1 or (len(path) == 2 and path[-1].token == self.end_token):
                            continue
                        # Merge the path (not including the first tuple that we're merging)
                        self.add(ignore_tuple, path[1:], graph=False)

                    # Add the node to a list to be removed later if it isn't an end node and doesn't have an end node in its children
                    if not child_node.value.is_end and not self.end_token in child_node.children:
                        # if not self.end_token in child_node.children:
                        to_remove.append(child_token)

                for child_token in to_remove:
                    node.children.pop(child_token)

            for token, neighbour in node.children.items():
                new_node, new_edge = neighbour
                if new_node.id_ not in visited:
                    visited.add(new_node.id_)
                    queue.append(neighbour)

    def search(self, tokens: List[str]) -> float:
        """Evaluate the activation on the first token in tokens"""
        current_tuple = self.trie_root

        # print("\n")
        activations = [0]

        for i, token in enumerate(reversed(tokens)):
            token = self.normalise(token)

            current_node, current_edge = current_tuple

            # print("i, token", i, [token])
            # print("current_node.children", current_node.children)

            if token in current_node.children or self.ignore_token in current_node.children:
                current_tuple = current_node.children[token] if token in current_node.children else \
                    current_node.children[self.ignore_token]

                node, edge = current_tuple
                # If the first token is not an activator, return early
                if i == 0:
                    if not node.value.activator:
                        break
                    activation = node.value.activation

                # print("node", node)

                if self.end_token in node.children:
                    # debug("Returning", activation)
                    end_node, _ = node.children[self.end_token]
                    end_activation = end_node.value.activation
                    activations.append(end_activation)

            else:
                break

        # Return the activation on the longest sequence
        return activations[-1]

    def forward(self, tokens_arr: List[List[str]], return_activations=True) -> List[List[float]]:
        if isinstance(tokens_arr[0], str):
            raise ValueError(f"tokens_arr must be of type List[List[str]]")

        # print("\n\n******PROCESSING*******")
        # print(tokens_arr)
        """Evaluate the activation on each token in some input tokens"""
        all_activations = []
        all_firings = []

        for tokens in tokens_arr:
            activations = []
            firings = []

            for j in range(len(tokens)):
                token_activation = self.search(tokens[:len(tokens) - j])
                activations.append(token_activation)
                firings.append(token_activation > self.activation_threshold)

            activations = list(reversed(activations))
            firings = list(reversed(firings))

            all_activations.append(activations)
            all_firings.append(firings)

            # print(list(zip(tokens, activations)))

        if return_activations:
            return all_activations
        return all_firings

    def build(self, start_node, graph=True):
        """Build a graph to visualise"""
        # print("\n\n******BUILDING*******")
        visited = set()  # List to keep track of visited nodes.
        queue = []  # Initialize a queue

        visited.add(start_node[0].id_)
        queue.append(start_node)

        zero_width = u'\u200b'
        # zero_width = "a"

        tokens_by_layer = {}
        node_id_to_graph_id = {}
        token_by_layer_count = defaultdict(Counter)
        added_ids = set()
        node_count = 0
        depth_to_subgraph = {}
        added_edges = set()

        node_edge_tuples = []

        adjust = lambda x, y: (x - y) / (1 - y)

        while queue:
            # print(queue)
            node, edge = queue.pop(0)

            node_edge_tuples.append((node, edge))

            for token, neighbour in node.children.items():
                # print("token", token)
                new_node, new_edge = neighbour
                if new_node.id_ not in visited:
                    visited.add(new_node.id_)
                    queue.append(neighbour)

        for node, edge in node_edge_tuples:
            token = node.value.token
            depth = node.depth

            # if token == "":
            #   continue

            if depth not in tokens_by_layer:
                tokens_by_layer[depth] = {}
                # depth_to_subgraph[depth] = Graph(name=f"cluster_{str(self.max_depth - depth)}")
                depth_to_subgraph[depth] = Digraph(name=f"cluster_{str(self.max_depth - depth)}")
                # depth_to_subgraph[depth].attr(label=f"Depth {str(depth)}")
                depth_to_subgraph[depth].attr(pencolor="white", penwidth="3")

            token_by_layer_count[depth][token] += 1

            if not graph:
                # This is a horrible hack to allow us to have a dict with the "same" token as multiple keys - by adding zero width spaces the tokens look the same but are actually different. This allows us to display a trie rather than a node-collapsed graph
                seen_count = token_by_layer_count[depth][token] - 1
                add = zero_width * seen_count
                token += add

            if token not in tokens_by_layer[depth]:
                tokens_by_layer[depth][token] = str(node_count)
                node_count += 1

            graph_node_id = tokens_by_layer[depth][token]
            node_id_to_graph_id[node.id_] = graph_node_id

            # for node, edge in reversed(node_edge_tuples):
            #   token = node.value.token
            #   depth = node.depth

            # graph_node_id = tokens_by_layer[depth][token]
            # node_id_to_graph_id[node.id_] = graph_node_id
            current_graph = depth_to_subgraph[depth]

            if depth == 0:
                # colour red according to activation for depth 0 tokens
                scaled_activation = int(adjust(node.value.activation, max(0, self.activation_threshold - 0.2)) * 255)
                rgb = (255, 255 - scaled_activation, 255 - scaled_activation)
            else:
                # colour blue according to importance for all other tokens
                # Shift and scale importance so the importance threshold becomes 0

                scaled_importance = int(adjust(node.value.importance, max(0.1, self.importance_threshold - 0.2)) * 255)
                rgb = (255 - scaled_importance, 255 - scaled_importance, 255)

            hex = "#{0:02x}{1:02x}{2:02x}".format(*self.clamp(rgb))

            # self.net.add_node(node.id_, label=node.value["token"], color=hex)

            if graph_node_id not in added_ids and not node.value.ignore:
                display_token = token.strip(zero_width)
                display_token = json.dumps(display_token).strip('[]"') if '"' not in token else display_token
                if set(display_token) == {" "}:
                    display_token = f"'{display_token}'"

                # self.net.node(graph_node_id, node.value["token"])
                # print("token", token, escape(token))
                fontcolor = "white" if depth != 0 and rgb[1] < 130 else "black"
                fontsize = "25" if len(display_token) < 12 else "18"
                edge_width = "7" if node.value.is_end else "3"
                # current_graph.node(
                #     graph_node_id, f"< <B> {escape(token)} </B> >", fillcolor=hex, shape="box",
                #     style="filled,solid", fontcolor=fontcolor, fontsize=fontsize,
                #     penwidth=edge_width
                # )

                current_graph.node(
                    graph_node_id, f"{escape(display_token)}", fillcolor=hex, shape="box",
                    style="filled,solid", fontcolor=fontcolor, fontsize=fontsize,
                    penwidth=edge_width
                )
                added_ids.add(graph_node_id)

            if edge.parent is not None and edge.parent.id_ in visited and not edge.parent.value.ignore:
                # self.net.add_edge(node.id_, edge.parent.id_, value=edge.weight, title=round(edge.weight, 2))
                # pprint(node_id_to_graph_id)
                # print([token])
                # print([edge.parent.value.token])
                # print([edge.parent.value.importance])
                graph_parent_id = node_id_to_graph_id[edge.parent.id_]
                # current_graph.edge(graph_parent_id, graph_node_id, constraint='false')
                edge_tuple = (graph_parent_id, graph_node_id)
                if edge_tuple not in added_edges:
                    self.net.edge(*edge_tuple, penwidth="3", dir="back")
                    added_edges.add(edge_tuple)

            # print("node", node)
            # print("edge", edge)
            # print("node.children", node.children)

            # for token, neighbour in node.children.items():
            #   # print("token", token)
            #   new_node, new_edge = neighbour
            #   if new_node.id_ not in visited:
            #     visited.add(new_node.id_)
            #     queue.append(neighbour)

        for depth, subgraph in depth_to_subgraph.items():
            self.net.subgraph(subgraph)

        path_parts = ['neuron_graphs', model_name]

        if self.folder_name is not None:
            path_parts.append(self.folder_name)

        path_parts.append(f"{self.layer}_{self.neuron}")

        save_path = base_path
        for path_part in path_parts:
            save_path += f"/{path_part}"
            if not os.path.exists(save_path):
                os.mkdir(save_path)

        self.net.format = 'svg'
        filename = "graph" if graph else "trie"
        self.net.render(f"{save_path}/{filename}", view=False)
        self.net.format = 'png'
        self.net.render(f"{save_path}/{filename}", view=False)
        # print(self.net.source)

    @staticmethod
    def clamp(arr):
        return [max(0, min(x, 255)) for x in arr]


class TokenPredictor:
    def __init__(self, model, layer, neuron, activation_threshold=0.5):
        self.model = model
        self.layer = layer
        self.neuron = neuron
        self.activation_threshold = activation_threshold

        self.layer_name = layer_index_to_name(layer)
        self.max_activation = activation_matrix[layer, neuron]

    def fit(self, texts):
        prepend_bos = False

        self.token_to_activations = defaultdict(list)
        for i, text in enumerate(texts):
            all_tokens = model.to_tokens(text, prepend_bos=prepend_bos)
            logits, cache = model.run_with_cache(all_tokens)
            neuron_activations = cache[self.layer_name][0, :, self.neuron]

            tokens = model.to_str_tokens(text, prepend_bos=prepend_bos)
            neuron_activations = neuron_activations.to("cpu")
            for token, activation in zip(tokens, neuron_activations):
                activation = activation.item()
                self.token_to_activations[token].append(activation / self.max_activation)

        self.token_to_activation = {token: np.max(activations) for token, activations in
                                    self.token_to_activations.items()}

    def forward(self, tokens_arr: List[List[str]], return_activations=True) -> List[List[float]]:
        all_activations = []
        all_firings = []

        for tokens in tokens_arr:
            activations = []
            firings = []

            for token in tokens:
                activation = self.token_to_activation.get(token, 0)

                activations.append(activation)
                firings.append(activation > self.activation_threshold)

            all_activations.append(activations)
            all_firings.append(firings)

        if return_activations:
            return all_activations
        return all_firings


import numpy as np


class NGramBaseline:
    def __init__(self, model, layer, neuron, prior_context=1, activation_threshold=0.5):
        self.model = model
        self.layer = layer
        self.neuron = neuron
        self.activation_threshold = activation_threshold
        self.prior_context = prior_context

        self.layer_name = layer_index_to_name(layer)
        self.max_activation = activation_matrix[layer, neuron]

    def fit(self, texts):
        prepend_bos = False

        self.seq_to_activations = defaultdict(list)
        self.activating_tokens = set()

        for i, text in enumerate(texts):
            all_tokens = model.to_tokens(text, prepend_bos=prepend_bos)
            logits, cache = model.run_with_cache(all_tokens)
            neuron_activations = cache[self.layer_name][0, :, self.neuron].cpu()

            tokens = model.to_str_tokens(text, prepend_bos=prepend_bos)
            for j, (token, activation) in enumerate(zip(tokens, neuron_activations)):
                activation = activation.item()
                if activation < self.activation_threshold:
                    continue
                token_seq = tokens[max(0, j - self.prior_context):j + 1]
                # print(token_seq, activation)
                self.activating_tokens.add(token)
                self.seq_to_activations["".join(token_seq)].append(activation / self.max_activation)

        self.seq_to_activation = {seq: np.max(activations) for seq, activations in self.seq_to_activations.items()}

        # pprint(self.seq_to_activation)

    def forward(self, tokens_arr: List[List[str]], return_activations=True) -> List[List[float]]:
        all_activations = []
        all_firings = []

        for tokens in tokens_arr:
            activations = []
            firings = []

            for j, token in enumerate(tokens):
                if token not in self.activating_tokens:
                    activations.append(0)
                    firings.append(0 > self.activation_threshold)
                    continue

                token_seq = tokens[max(0, j - self.prior_context):j + 1]
                activation = self.seq_to_activation.get("".join(token_seq), 0)

                activations.append(activation)
                firings.append(activation > self.activation_threshold)

            all_activations.append(activations)
            all_firings.append(firings)

        if return_activations:
            return all_activations
        return all_firings


import random
import time


def run_training(layers, neurons, folder_name, sample_num=None, params=None, start_neuron=None):
    if params is None or not params:
        params = {
            "importance_threshold": 0.75,
            "n": 5,
            "max_train_size": None,
            "train_proportion": 0.5,
            "max_eval_size": 0.5,
            "activation_threshold": 0.5,
            "token_activation_threshold": 1,
            "fire_threshold": 0.5
        }
    print(f"{params=}\n")
    random.seed(0)

    all_neuron_indices = [i for i in range(neurons)]

    if not os.path.exists(f"{base_path}/neuron_graphs/{model_name}"):
        os.mkdir(f"{base_path}/neuron_graphs/{model_name}")

    neuron_store = NeuronStore(f"{base_path}/neuron_graphs/{model_name}/neuron_store.json")

    folder_path = os.path.join(base_path, f"neuron_graphs/{model_name}/{folder_name}")

    if not os.path.exists(folder_path):
        print("Making", folder_path)
        os.mkdir(folder_path)

    if os.path.exists(f"{folder_path}/stats.json"):
        with open(f"{folder_path}/stats.json") as ifh:
            all_stats = json.load(ifh)

    else:
        all_stats = {}

    printerval = 10

    for layer in layers:
        t1 = time.perf_counter()

        if sample_num is None:
            chosen_neuron_indices = all_neuron_indices
        else:
            chosen_neuron_indices = random.sample(all_neuron_indices, sample_num)
            chosen_neuron_indices = sorted(chosen_neuron_indices)

        all_stats[layer] = {}
        for i, neuron in enumerate(chosen_neuron_indices):
            if start_neuron is not None and neuron < start_neuron:
                continue

            print(f"{layer=} {neuron=}")
            try:
                stats = train_and_eval(model, layer, neuron, fast_aug, folder_name=folder_name, neuron_store=neuron_store,
                                       **params)

                all_stats[layer][neuron] = stats

                if i % printerval == 0:
                    t2 = time.perf_counter()
                    elapsed = t2 - t1
                    rate = printerval / elapsed
                    remaining = (len(chosen_neuron_indices) - i) / rate / 60
                    print(
                        f"{i} complete, batch took {elapsed / 60:.2f} mins, {rate=:.2f} neurons/s, {remaining=:.1f} mins")
                    t1 = t2

                    neuron_store.save()
                    with open(f"{folder_path}/stats.json", "w") as ofh:
                        json.dump(all_stats, ofh, indent=2)

            except Exception as e:
                print(e)
                print("Failed")

    neuron_store.save()
    with open(f"{folder_path}/stats.json", "w") as ofh:
        json.dump(all_stats, ofh, indent=2)


if __name__ == "__main__":
    """
    Instructions:
    Download word_to_casings.json from the repo and put in data/
    Set layer_ending to "mlp.hook_mid" for SoLU models and "mlp.hook_post" for GeLU models
    Download from the repo or scrape (with scrape.py) the activation matrix for the model and put in data/
    Set model_name to the name of the model you want to run for
    Set the parameters in the run section as desired
    Run this file!    
    
    It will create neuron graphs for the specified layers and neurons in neuron_graphs/model_name/folder_name
    It'll also save the stats for each neuron in neuron_graphs/model_name/folder_name/stats.json
    And it will save the neuron store in neuron_graphs/model_name/neuron_store.json
    """

    # Set these as desired
    model_name = "gpt2-small"
    layer_ending = "mlp.hook_post"

    # ================ Setup ================
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = HookedTransformer.from_pretrained(model_name).to(device)

    base_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__))))

    # Save the activation matrix for the model to data/
    with open(os.path.join(base_path, f"data/activation_matrix-{model_name}.json")) as ifh:
        activation_matrix = json.load(ifh)
        activation_matrix = np.array(activation_matrix)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    aug_model_checkpoint = "distilbert-base-uncased"
    aug_model = AutoModelForMaskedLM.from_pretrained(aug_model_checkpoint).to(device)
    aug_tokenizer = AutoTokenizer.from_pretrained(aug_model_checkpoint)

    stick_tokens = {"'"}
    word_tokenizer = WordTokenizer(set(), stick_tokens)
    fast_aug = FastAugmenter(aug_model, aug_tokenizer, word_tokenizer, model)

    with open(f"{base_path}/data/word_to_casings.json") as ifh:
        word_to_casings = json.load(ifh)

    # main()

    if not os.path.exists(f"{base_path}/neuron_graphs"):
        os.mkdir("neuron_graphs")

    # ================ Run ================
    # Run training for the specified layers and neurons

    folder_name = "layer_0"

    # Override params as desired - sensible defaults are set in run_training
    params = {}

    run_training(
        # List of layers to run for
        layers=[0],
        # Number of neurons in each layer
        neurons=3072,
        # Neuron to start at (useful for resuming - None to start at 0)
        start_neuron=None,
        # Folder to save results in
        folder_name=folder_name,
        # Number of neurons to sample from each layer (None for all neurons)
        sample_num=None,
        params=params
    )

    get_summary_stats(f"{base_path}/neuron_graphs/{model_name}/{folder_name}/stats.json")
