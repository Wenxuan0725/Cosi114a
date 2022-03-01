# hw3.py
# Version 1.2
# 10/28/2021

import collections
import math
import random
from collections import defaultdict, Counter
from math import log
from typing import Sequence, Iterable, Generator

############################################################
# The following constants and function are provided as helpers.
# Do not modify them! The stubs for what you need to implement are later in the file.

# DO NOT MODIFY
random.seed(0)

# DO NOT MODIFY
START_TOKEN = "<start>"
# DO NOT MODIFY
END_TOKEN = "<end>"
# DO NOT MODIFY
NEG_INF = float("-inf")


def load_tokenized_file(path: str) -> Generator[Sequence[str], None, None]:
    """Yield sentences as sequences of tokens."""
    with open(path, encoding="utf8") as file:
        for line in file:
            line = line.rstrip("\n")
            tokens = line.split(" ")
            yield tuple(tokens)


def sample(probs: dict[str, float]) -> str:
    """Return a sample from a distribution."""
    # To avoid relying on the dictionary iteration order, sort items
    items = sorted(probs.items())
    # Now split them back up into keys and values
    keys, vals = zip(*items)
    # Choose using the weighs in the values
    return random.choices(keys, weights=vals)[0]


############################################################
# The stubs below this are the ones that you should fill in.
# Do not modify anything above this line other than to add any needed imports.


def bigrams(sentence: Sequence[str]) -> list[tuple[str, str]]:
    d = list()
    start = START_TOKEN
    end = END_TOKEN
    for i in sentence:
        tup = (start, i)
        d.append(tup)
        start = i
    tup = (start, end)
    d.append(tup)
    return d

def trigrams(sentence: Sequence[str]) -> list[tuple[str, str, str]]:
    d = list()
    start1 = START_TOKEN
    start2 = START_TOKEN
    end1 = END_TOKEN
    end2 = END_TOKEN
    for i in sentence:
        tup = (start1, start2, i)
        d.append(tup)
        start1 = start2
        start2 = i
    tup = (start1, start2, end1)
    d.append(tup)
    tup = (start2, end1, end2)
    d.append(tup)
    return d

def count_bigrams(sentences: Iterable[Sequence[str]]) -> Counter[tuple[str,str]]:
    c:Counter[tuple[str,str]] = Counter()
    for item in sentences:
        c.update(bigram for bigram in bigrams(item))
    return c

def count_trigrams(sentences: Iterable[Sequence[str]]) -> Counter[tuple[str,str,str]]:
    c:Counter[tuple[str,str,str]] = Counter()
    for item in sentences:
        c.update(trigram for trigram in trigrams(item))
    return c

def bigram_probs(
    sentences: Iterable[Sequence[str]],
) -> dict[str, dict[str, float]]:
    data = count_bigrams(sentences)
    res = defaultdict(dict[str, float])
    temp = collections.defaultdict(Counter)
    for item in data:
        temp[item[0]][item[1]] += data[item]
    for key,val in temp.items():
        current_sum = sum(val.values())
        part1 = collections.defaultdict()
        for items in val:
            part1[items] = float(val[items]/current_sum)
        part1 = dict(part1)
        res[key] = part1
    res1 = dict(res)
    return res1



def trigram_probs(
    sentences: Iterable[Sequence[str]],
) -> dict[tuple[str, str], dict[str, float]]:
    data = count_trigrams(sentences)
    res = defaultdict(dict[str, float])
    temp = collections.defaultdict(Counter)
    for item in data:
        record = (item[0], item[1])
        temp[record][item[2]] += data[item]
    for key, val in temp.items():
        current_sum = sum(val.values())
        part1 = collections.defaultdict()
        for items in val:
            part1[items] = float(val[items] / current_sum)
        part1 = dict(part1)
        res[key] = part1
    res1 = dict(res)
    return res1


def sample_bigrams(probs: dict[str, dict[str, float]]) -> list[str]:
    current = START_TOKEN
    res = list()
    while current != END_TOKEN:
        data = sample(probs[current])
        if data != END_TOKEN:
            res.append(data)
        current = data
    return res



def sample_trigrams(probs: dict[tuple[str, str], dict[str, float]]) -> list[str]:
    current = (START_TOKEN, START_TOKEN)
    res = list()
    while current != (END_TOKEN,END_TOKEN):
        data = sample(probs[current])
        if data != END_TOKEN:
            res.append(data)
        else:
            return res
        current = (current[1], data)
    return res


def bigram_sequence_prob(
    sequence: Sequence[str], probs: dict[str, dict[str, float]]
) -> float:
    data = list(sequence)
    data.insert(0, START_TOKEN)
    data.insert(len(data), END_TOKEN)
    a = 0.0
    for i in range(0, len(data)-1):
        temp = probs.get(data[i])
        if data[i+1] not in temp:
            return NEG_INF
        else:
            a += math.log(temp.get(data[i+1]))
    return a




def trigram_sequence_prob(
    sequence: Sequence[str], probs: dict[tuple[str, str], dict[str, float]]
) -> float:
    data = list(sequence)
    data.insert(0, START_TOKEN)
    data.insert(0, START_TOKEN)
    data.insert(len(data), END_TOKEN)
    data.insert(len(data), END_TOKEN)
    a = 0.0
    for i in range(0, len(data) - 2):
        temp = probs.get((data[i], data[i+1]))
        if data[i + 2] not in temp:
            return NEG_INF
        else:
            a += math.log(temp.get(data[i + 2]))
    return a
