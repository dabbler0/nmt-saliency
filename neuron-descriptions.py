from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer
from urlparse import urlparse
import json
import numpy
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

saliencies_file = sys.argv[1]
out_file = sys.argv[2]

# Load all data

descriptions = []
with open(saliencies_file) as f:
    i = 0
    for line in f:
        i += 1
        if i > 1000:
            break
        try:
            descriptions.append(json.loads(line.replace('nan,', 'NaN,').replace('nan]', 'NaN]')))
        except Exception as e:
            print('An exception occurred while parsing:')
            print(e)
            continue

print('Done reading.')

def describe_neuron_by_max(neuron):
    methods = {
        'sgrad': {},
        'lime': {},
        'lrp': {},
        'erasure': {},
    }

    # For each line,
    for line in descriptions:
        desc = line['description']['en-es-0']['saliencies']
        text = line['line']
        index = line['index']

        tokens = ['{START}}'] + text.split(' ')[:index-1]

        # See which tokens are on this line
        present_tokens = set(tokens)

        # For each method,
        for method in methods:

            # For all the present tokens, increment their appearance number
            for t in present_tokens:
                # Initialize to zero if it's not here
                if t not in methods[method]:
                    methods[method][t] = [0, 0, 0]
                # Otherwise increment
                methods[method][t][2] += 1

            # Determine the most salient token in the positive/negative direction
            most_salient_positive = max(range(index), key = lambda x: desc[method][x][neuron])
            most_salient_negative = min(range(index), key = lambda x: desc[method][x][neuron])

            # Increment
            if desc[method][most_salient_positive][neuron] > 0:
                methods[method][tokens[most_salient_positive]][0] += 1
            if desc[method][most_salient_negative][neuron] < 0:
                methods[method][tokens[most_salient_negative]][1] += 1

    # Normalize
    new_dict = {}
    for method in methods:
        new_array = []
        for recognized_token in methods[method]:
            if methods[method][recognized_token][2] > 10:
                methods[method][recognized_token] = (
                    methods[method][recognized_token][0] / (methods[method][recognized_token][2] + 1.), # (Smoothing)
                    methods[method][recognized_token][1] / (methods[method][recognized_token][2] + 1.)
                )

                if methods[method][recognized_token] != (0, 0):
                    new_array.append((recognized_token.__repr__(), methods[method][recognized_token]))

        new_dict[method] = sorted(new_array, key = lambda x: -max(x[1][0], x[1][1]))

    return new_dict


smoothing_factor = 10
def describe_neuron_by_mean(neuron):
    methods = {
        'sgrad': {},
        'lime': {},
        'lrp': {},
        'erasure': {},
    }

    activations = {}

    # For each line,
    for line in descriptions:
        desc = line['description']['en-es-0']['saliencies']
        text = line['line']
        index = line['index']

        tokens = ['{START}}'] + text.split(' ')[:index-1]

        for i, token in enumerate(tokens):
            for method in methods:
                if token not in methods[method]:
                    methods[method][token] = [[0, smoothing_factor], [0, smoothing_factor]] # Smoothing factor
                act = desc[method][i][neuron]
                if act > 0:
                    methods[method][token][0][0] += act
                    methods[method][token][0][1] += 1
                elif act < 0:
                    methods[method][token][1][0] += act
                    methods[method][token][1][1] += 1

            # Special "method": activations
            if token not in activations:
                activations[token] = [[0, smoothing_factor], [0, smoothing_factor]]
            act = line['description']['en-es-0']['activations'][-1][neuron]
            if act > 0:
                activations[token][0][0] += act
                activations[token][0][1] += 1
            elif act < 0:
                activations[token][1][0] += act
                activations[token][1][1] += 1

    methods['activation'] = activations

    for method in methods:
        for token in methods[method]:
            if methods[method][token][0][1] != 0:
                methods[method][token][0] = methods[method][token][0][0] / methods[method][token][0][1]
            else:
                methods[method][token][0] = 0

            if methods[method][token][1][1] != 0:
                methods[method][token][1] = methods[method][token][1][0] / methods[method][token][1][1]
            else:
                methods[method][token][1] = 0

    for method in methods:
        l = []
        '''
        for token in methods[method]:
            l.append((token, methods[method][token][0], methods[method][token][1]))

        methods[method] = sorted(l, key = lambda x: -max(abs(x[1]), abs(x[2])))
        '''

        for token in methods[method]:
            l.append((token.__repr__(), max(abs(methods[method][token][0]), abs(methods[method][token][1]))))

        methods[method] = sorted(l, key = lambda x: -x[1])

    return methods

print('Loaded.')

#neuron = int(sys.argv[1])

# Try describing the neuron
#f = open('/data/sls/scratch/abau/test%d.json' % (neuron,), 'w')

big_dictionary = {
    'sgrad': {},
    'lime': {},
    'lrp': {},
    'erasure': {},
    'activation': {}
}

for neuron in range(500):
    description = describe_neuron_by_mean(neuron)

    print('NEURON', neuron)

    # Determine the "interpretability" score, which we take to mean
    # the mean square distance to your quintile
    for method in description:
        mean = 0.0
        for token in description[method]:
            mean += token[1]
        mean /= len(description[method])

        lower_quintile = 0
        lower_ql = 0
        upper_quintile = 0
        upper_ql = 0

        upper_quintile_tokens = []

        for token in description[method]:
            # Upper quintile
            if token[1] > mean:
                upper_quintile_tokens.append(token[0])
                upper_quintile += token[1]
                upper_ql += 1

            # Lower quintile
            else:
                lower_quintile += token[1]
                lower_ql += 1

        score = 0
        for token in description[method]:
            # Distance to quintile
            if token[1] > mean:
                score += (token[1] - upper_quintile) ** 2
            else:
                score += (token[1] - lower_quintile) ** 2

        score /= len(description[method])

        print('%s %f' % (method, score))
        print('%s ATTENTION TOKENS:' % (method,))
        print('\n  '.join(upper_quintile_tokens[:10]))

        big_dictionary[method][neuron] = {
            'score': score,
            'saliencies': description[method]
        }

    #json.dump(description, f, indent=2)
    print('Described.')

f = open(out_file, 'w')
json.dump(big_dictionary, f)
