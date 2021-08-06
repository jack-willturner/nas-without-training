import torch.nn as nn
import numpy as np


def fish(input, target, network):
    criterion = nn.CrossEntropyLoss()

    _, _, ints = network(input, get_ints=True)

    for int in ints:
        int.retain_grad()

    _ = criterion.logits, target)

    fishers = []

    for int in ints:
        go_fish = (int.data.detach() * int.grad.detach()).sum(-1).sum(-1).pow(2).mean(0).sum()
        fishers.append(go_fish)
    
    return sum(fishers)


def hooklogdet(K, labels=None):
    s, ld = np.linalg.slogdet(K)
    return ld

def random_score(jacob, label=None):
    return np.random.normal()


_scores = {
        'hook_logdet': hooklogdet,
        'random': random_score,
        'fish': fish,
        }

def get_score_func(score_name):
    return _scores[score_name]
