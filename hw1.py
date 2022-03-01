from collections import Counter, defaultdict

from typing import Iterable, TypeVar, Sequence

# DO NOT MODIFY
T = TypeVar("T")

# DO NOT MODIFY
START_TOKEN = "<start>"
# DO NOT MODIFY
END_TOKEN = "<end>"


def counts_to_probs(counts: Counter[T]) -> defaultdict[T, float]:
    dict(counts)
    d = defaultdict(float)
    sum1 = sum(counts.values())
    for i in counts:
        counts[i] /= sum1
        d[i] = counts[i]
    return d


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


def count_unigrams(sentences: Iterable[list[str]], lower: bool = False) -> Counter[str]:
    c = Counter()
    for item in sentences:
        if lower:
            c.update(i.lower() for i in item)
        else:
            c.update(item)
    return c


def count_bigrams(
        sentences: Iterable[list[str]], lower: bool = False
) -> Counter[tuple[str, str]]:
    c = Counter()
    for item in sentences:
        if lower:
            c.update([tuple([i.lower() for i in bigram]) for bigram in bigrams(item)])
        else:
            c.update([bigram for bigram in bigrams(item)])
    return c


def count_trigrams(
        sentences: Iterable[list[str]], lower: bool = False
) -> Counter[tuple[str, str, str]]:
    c = Counter()
    for item in sentences:
        if lower:
            c.update([tuple([i.lower() for i in trigram]) for trigram in trigrams(item)])
        else:
            c.update([trigram for trigram in trigrams(item)])
    return c
