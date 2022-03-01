# hw4.py
# Version 1.0
# 11/2/2021


from abc import abstractmethod, ABC
from collections import Counter, defaultdict
from math import log
from operator import itemgetter
from typing import Any, Generator, Iterable, Sequence

############################################################
# The following constants, classes, and function are provided as helpers.
# Do not modify them! The stubs for what you need to implement are later in the file.

# DO NOT MODIFY
NEG_INF = float("-inf")


# DO NOT MODIFY
class Token:
    """Stores the text and tag for a token.

    Hashable and cleaner than indexing tuples all the time.
    """

    def __init__(self, token: str, tag: str):
        self.text = token
        self.tag = tag

    def __str__(self):
        return f"{self.text}/{self.tag}"

    def __repr__(self):
        return f"<Token {str(self)}>"

    def __eq__(self, other: Any):
        return (
                isinstance(other, Token) and self.text == other.text and self.tag == other.tag
        )

    def __lt__(self, other: "Token"):
        return self.to_tuple() < other.to_tuple()

    def __hash__(self):
        return hash(self.to_tuple())

    def to_tuple(self):
        """Return the text and tag as a tuple.

        Example:
        >>> token = Token("apple", "NN")
        >>> token.to_tuple()
        ('apple', 'NN')
        """
        return self.text, self.tag

    @staticmethod
    def from_tuple(t: tuple[str, ...]):
        """
        Creates a Token object from a tuple.
        """
        assert len(t) == 2
        return Token(t[0], t[1])

    @staticmethod
    def from_string(s: str) -> "Token":
        """Create a Token object from a string with the format 'token/tag'.

        Sample usage: Token.from_string("cat/NN")
        """
        return Token(*s.rsplit("/", 1))


# DO NOT MODIFY
class Tagger(ABC):
    @abstractmethod
    def train(self, sentences: Iterable[Sequence[Token]]) -> None:
        """Train the part of speech tagger by collecting needed counts from sentences."""
        raise NotImplementedError

    @abstractmethod
    def tag_sentence(self, sentence: Sequence[str]) -> list[str]:
        """
        Tags a sentence with part of speech tags.
        Sample usage:
            tag_sentence(["I", "ate", "an", "apple"])
             returns: ["PRP", "VBD", "DT", "NN"]
        """
        raise NotImplementedError

    def tag_sentences(
            self, sentences: Iterable[Sequence[str]]
    ) -> Generator[list[str], None, None]:
        """
        Tags each sentence's tokens with part of speech tags and
        yields the corresponding list of part of speech tags.
        """
        for sentence in sentences:
            yield self.tag_sentence(sentence)

    def test(
            self, tagged_sents: Iterable[Sequence[Token]]
    ) -> tuple[list[str], list[str]]:
        """
        Runs the tagger over all the sentences and returns a tuple with two lists:
        the predicted tag sequence and the actual tag sequence.
        The predicted and actual tags can then be used for calculating accuracy or other
        metrics.
        This does not preserve sentence boundaries.
        """
        predicted: list[str] = []
        actual: list[str] = []
        for sent in tagged_sents:
            predicted.extend(self.tag_sentence([t.text for t in sent]))
            actual.extend([t.tag for t in sent])
        return predicted, actual


# DO NOT MODIFY
def _safe_log(n: float) -> float:
    """Return the log of a number or -inf if the number is zero."""
    return NEG_INF if n == 0.0 else log(n)


# DO NOT MODIFY
def _max_item(scores: dict[str, float]) -> tuple[str, float]:
    """Return the key and value with the highest value."""
    # PyCharm gives a false positive type error here
    # noinspection PyTypeChecker
    return max(scores.items(), key=itemgetter(1))


# DO NOT MODIFY
def _most_frequent_item(counts: Counter[str]) -> str:
    """Return the most frequent item in a Counter."""
    assert counts, "Counter is empty"
    top_item, _ = counts.most_common(1)[0]
    return top_item


############################################################
# The stubs below this are the ones that you should fill in.
# Do not modify anything above this line other than to add any needed imports.


class MostFrequentTagTagger(Tagger):
    def __init__(self):
        # Add an attribute to store the most frequent tag
        self.default_tag = ""

    def train(self, sentences: Iterable[Sequence[Token]]) -> None:
        tag_counter = Counter()
        for sentence in sentences:
            for token in sentence:
                tag_counter[token.tag] += 1
        self.default_tag = _most_frequent_item(tag_counter)

    def tag_sentence(self, sentence: Sequence[str]) -> list[str]:
        res = [self.default_tag] * len(sentence)
        return res


class UnigramTagger(Tagger):
    def __init__(self):
        self.default_tag = defaultdict()
        self.most_default_tag = ""
        # Add Counters/dicts/defaultdicts/etc. that you need here.

    def train(self, sentences: Iterable[Sequence[Token]]):
        tag_data = defaultdict(Counter)
        for sentence in sentences:
            for token in sentence:
                tag_data[token.text][token.tag] += 1
        max_count = 0
        for key, value in tag_data.items():
            self.default_tag[key] = _max_item(dict(value))[0]
            if max_count < _max_item(dict(value))[1]:
                self.most_default_tag = _max_item(dict(value))[0]

    def tag_sentence(self, sentence: Sequence[str]) -> list[str]:
        res = list()
        for word in sentence:
            if word in self.default_tag:
                res.append(self.default_tag[word])
            else:
                res.append(self.most_default_tag)
        return res


class SentenceCounter:
    def __init__(self, k: float):
        self.k = k
        self.tag_record = defaultdict(Counter)
        self.tag_set = set()
        self.total_tag = 0
        self.tag_sum = Counter()
        self.tag_unique_word = defaultdict(set)
        self.initial_tag = Counter()
        self.total_initial_tag = 0
        self.transition_count = Counter()
        self.transition_prev = Counter()
        self.tag_set_list = list()
        # Add Counters/dicts/defaultdicts/etc. that you need here.

    def count_sentences(self, sentences: Iterable[Sequence[Token]]) -> None:
        """Count token text and tags in sentences.

        After this function runs the SentenceCounter object should be ready
        to return values for initial, transition, and emission probabilities
        as well as return the sorted tagset.
        """
        for sentence in sentences:
            initial = True
            prev_tag = ""
            for token in sentence:
                if initial:
                    self.initial_tag[token.tag] += 1
                    self.total_initial_tag += 1
                    initial = False
                    prev_tag = token.tag
                else:
                    self.transition_count[(prev_tag, token.tag)] += 1
                    self.transition_prev[prev_tag] += 1
                    prev_tag = token.tag
                self.tag_record[token.tag][token.text] += 1
                self.tag_set.add(token.tag)
                self.tag_unique_word[token.tag].add(token.text)
                self.total_tag += 1
        for key in self.tag_record:
            self.tag_sum[key] = sum(self.tag_record[key].values())
        self.tag_set_list = list(self.tag_set)
        self.tag_set_list.sort()

    def conditional_tag_count(self, tag: str, word: str) -> int:
        if tag not in self.tag_record:
            return 0
        if word not in self.tag_record[tag]:
            return 0
        return self.tag_record[tag][word]

    def tagset(self) -> list[str]:
        return self.tag_set_list

    def emission_prob(self, tag: str, word: str) -> float:
        if tag not in self.tag_record:
            return float(0)
        numerator = float(self.conditional_tag_count(tag, word) + self.k)
        denominator = float(self.tag_sum[tag] + self.k * len(self.tag_unique_word[tag]))
        if denominator == 0:
            return float(0)
        else:
            prob_w = numerator / denominator
            return prob_w

    def transition_prob(self, prev_tag: str, current_tag: str) -> float:
        if (prev_tag, current_tag) not in self.transition_count:
            return float(0)
        elif prev_tag not in self.transition_prev:
            return float(0)
        return float(self.transition_count[(prev_tag, current_tag)] / self.transition_prev[prev_tag])

    def initial_prob(self, tag: str) -> float:
        if self.initial_tag[tag] == 0:
            return float(0)
        return float(self.initial_tag[tag] / self.total_initial_tag)


class BigramTagger(Tagger, ABC):
    # You can add additional methods to this class if you want to share anything
    # between the greedy and Viterbi taggers. However, do not modify any of the
    # implemented methods.
    def __init__(self, k: float) -> None:
        # DO NOT MODIFY THIS METHOD
        self.counter = SentenceCounter(k)

    def train(self, sents: Iterable[Sequence[Token]]) -> None:
        # DO NOT MODIFY THIS METHOD
        self.counter.count_sentences(sents)

    def sequence_probability(self, sentence: Sequence[str], tags: Sequence[str]) -> float:
        prob = float(
            _safe_log(self.counter.initial_prob(tags[0])) + _safe_log(self.counter.emission_prob(tags[0], sentence[0])))
        for i in range(1, len(tags)):
            prob += float(_safe_log(self.counter.transition_prob(tags[i - 1], tags[i])) + _safe_log(
                self.counter.emission_prob(tags[i], sentence[i])))
        return prob


class GreedyBigramTagger(BigramTagger):
    # DO NOT DEFINE __init__ or train

    def tag_sentence(self, sentence: Sequence[str]) -> list[str]:
        record = defaultdict()
        words = list()
        tags = list()
        for word in sentence:
            words.append(word)
            for tag in self.counter.tag_set_list:
                temp_tags = tags.copy()
                temp_tags.append(tag)
                temp_prob = self.sequence_probability(words, temp_tags)
                record[tag] = temp_prob
            tags.append(_max_item(record)[0])
        return tags


class ViterbiBigramTagger(BigramTagger):
    # DO NOT DEFINE __init__ or train

    def tag_sentence(self, sentence: Sequence[str]) -> list[str]:
        back_pointers = [{} for _ in range(len(sentence))]
        best_scores = [{} for _ in range(len(sentence))]
        count = 0
        for tag in self.counter.tag_set_list:
            best_scores[count][tag] = _safe_log(self.counter.initial_prob(tag)) + _safe_log(
                self.counter.emission_prob(tag, sentence[count]))
        count += 1
        for word_index in range(1, len(sentence)):
            for tag in self.counter.tag_set_list:
                score_record = dict()
                for prev_tag in self.counter.tag_set_list:
                    score_record[prev_tag] = best_scores[count - 1][prev_tag] + _safe_log(
                        self.counter.emission_prob(tag, sentence[count])) + _safe_log(
                        self.counter.transition_prob(prev_tag, tag))
                best_scores[count][tag] = _max_item(score_record)[1]
                back_pointers[count][tag] = _max_item(score_record)[0]
            count += 1
        res = list()
        res.append(_max_item(best_scores[count - 1])[0])
        for score_index in range(len(sentence) - 1, 0, -1):
            best_tag = back_pointers[score_index][_max_item(best_scores[score_index])[0]]
            res.append(best_tag)
        res.reverse()
        return res
