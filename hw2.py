# hw2.py
# Version 1.0
# 9/28/2021
import collections
import json
from collections import Counter

import math
from typing import (
    Iterable,
    Any,
    Sequence,
    Generator,
)

############################################################
# The following classes and methods are provided as helpers.
# Do not modify them! The stubs for what you need to implement are later in the file.

# DO NOT MODIFY
START_TOKEN = "<start>"
# DO NOT MODIFY
END_TOKEN = "<end>"


# DO NOT MODIFY
class AirlineSentimentInstance:
    """Represents a single data point from the airline sentiment dataset.

    Each data point consists of
    - the airline (str)
    - the sentiment label (str)
    - the review itself (list of lists)
        - outer lists represent sentences
        - inner lists represent tokens within sentences
    """

    def __init__(self, label: str, sentences: list[list[str]], airline: str) -> None:
        self.label = label
        self.sentences = sentences
        self.airline = airline

    def __repr__(self) -> str:
        return f"label={self.label}; sentences={self.sentences}"

    def __str__(self) -> str:
        return self.__repr__()

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "sentences": self.sentences,
            "airline": self.airline,
        }

    @classmethod
    def from_dict(cls, json_dict: dict[str, Any]) -> "AirlineSentimentInstance":
        return AirlineSentimentInstance(
            json_dict["label"], json_dict["sentences"], json_dict["airline"]
        )


# DO NOT MODIFY
class SentenceSplitInstance:
    """Represents a potential sentence boundary within a given string.

    An instance may correspond to a true boundary or not.
    The boundary is represented using the following properties:
    - label (str): either 'y' (true) or 'n' (false)
    - left_context (str): token immediately preceding the sentence boundary token
    - token (str): string representing the sentence boundary (str)
        - for example, a period (.) or question mark (?)
        - the last token of the sentence if this is a true sentence boundary.
    - right_context (str): token immediately following the sentence boundary token
    """

    def __init__(
        self, label: str, left_context: str, token: str, right_context: str
    ) -> None:
        self.label = label
        self.left_context = left_context
        self.token = token
        self.right_context = right_context

    def __repr__(self) -> str:
        return " ".join(
            [
                f"label={self.label};",
                f"left={self.left_context};",
                f"token={self.token};",
                f"right={self.right_context}",
            ]
        )

    def __str__(self) -> str:
        return self.__repr__()

    def to_dict(self):
        return {
            "label": self.label,
            "left": self.left_context,
            "token": self.token,
            "right": self.right_context,
        }

    @classmethod
    def from_dict(cls, json_dict: dict[Any, Any]) -> "SentenceSplitInstance":
        return SentenceSplitInstance(
            json_dict["label"],
            json_dict["left"],
            json_dict["token"],
            json_dict["right"],
        )


# DO NOT MODIFY
def load_airline_instances(
    datapath: str,
) -> Generator[AirlineSentimentInstance, None, None]:
    with open(datapath, encoding="utf8") as infile:
        json_list = json.load(infile)

        for json_item in json_list:
            yield AirlineSentimentInstance.from_dict(json_item)


# DO NOT MODIFY
def load_segmentation_instances(
    datapath: str,
) -> Generator[SentenceSplitInstance, None, None]:
    with open(datapath, encoding="utf8") as infile:
        json_list = json.load(infile)

        for json_item in json_list:
            yield SentenceSplitInstance.from_dict(json_item)


# DO NOT MODIFY
class ClassificationInstance:
    def __init__(self, label: str, features: list[str]) -> None:
        self.label = label
        self.features = features


############################################################
# The stubs below this are the ones that you should fill in.
# Do not modify anything above this line other than to add any needed imports.



def accuracy(predictions: Sequence[str], expected: Sequence[str]) -> float:
    if not predictions or not expected or len(predictions) != len(expected):
        raise ValueError("predictions and excepted do not have the same length.")
    else:
        t = 0.0
        f = 0.0
        for (predict, expect) in zip(predictions, expected):
            # predictions[i] = predictions[i].lower()
            # expected[i] = expected[i].lower()
            if predict.lower() == expect.lower():
                t += 1
            else:
                f += 1
        if t+f == 0:
            return 0
        return float(t/(t+f))


def recall(predictions: Sequence[str], expected: Sequence[str], label: str) -> float:
    if not predictions or not expected or len(predictions)!= len(expected):
        raise ValueError("predictions and excepted do not have the same length.")
    else:
        label = label.lower()
        tp = 0.0
        fn = 0.0
        for (prediction, expect) in zip(predictions, expected):
            if prediction.lower() == expect.lower() == label:
                tp += 1
            elif prediction.lower() != label and expect.lower() == label:
                fn += 1
        if tp + fn == 0:
            return 0.0
        return float(tp / (tp+fn))


def precision(predictions: Sequence[str], expected: Sequence[str], label: str) -> float:
    if not predictions or not expected or len(predictions)!= len(expected):
        raise ValueError("predictions and excepted do not have the same length.")
    else:
        label = label.lower()
        tp = 0.0
        fp = 0.0
        for (prediction, expect) in zip(predictions, expected):
            # predictions[i] = predictions[i].lower()
            # expected[i] = expected[i].lower()
            if prediction.lower() == expect.lower() == label:
                tp += 1
            elif prediction.lower() == label and expect.lower() != label:
                fp += 1
        if tp + fp == 0:
            return 0.0
        return float(tp / (tp + fp))


def f1(predictions: Sequence[str], expected: Sequence[str], label: str) -> float:
    label = label.lower()
    r = recall(predictions, expected, label)
    p = precision(predictions, expected, label)
    if p+r == 0:
        return 0.0
    return float(2*(p*r)/(p+r))


class UnigramAirlineSentimentFeatureExtractor:
    def extract_features(
        self, instance: AirlineSentimentInstance
    ) -> ClassificationInstance:
        res = set()
        for i in instance.sentences:
            for j in i:
                j = j.lower()
                res.add(j)
        res = sorted(res)
        data = list(res)
        result = ClassificationInstance(instance.label, data)
        return result


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


class BigramAirlineSentimentFeatureExtractor:
    def extract_features(
        self, instance: AirlineSentimentInstance
    ) -> ClassificationInstance:
        res = set()
        data = instance.sentences
        for i in data:
            for j in range(0, len(i)):
                i[j] = i[j].lower()
            temp = bigrams(i)
            for k in temp:
                res.add(str(k).lower())
        res = sorted(res)
        data = list(res)
        result = ClassificationInstance(instance.label, data)
        return result

class BaselineSegmentationFeatureExtractor:
    def extract_features(self, instance: SentenceSplitInstance) -> ClassificationInstance:
        res = list()
        res.append("split_tok="+instance.token)
        res.append("right_tok="+instance.right_context)
        res.append("left_tok="+instance.left_context)
        result = ClassificationInstance(instance.label,res)
        return result


class InstanceCounter:
    def __init__(self) -> None:
        self.label_counts = Counter()
        self.count = 0
        self.features = Counter()
        self.set_features = set()
        self.features_data = collections.defaultdict(Counter)
        self.features_label = Counter()

    def count_instances(self, instances: Iterable[ClassificationInstance]) -> None:
        for i in instances:
            self.features.update(i.features)
            self.features_data[i.label].update(i.features)
            self.label_counts.update([i.label])#这样可以，但是可以写成self.label_counts[i.label]+=1, counter本质上还是dict
            self.count += 1
            self.set_features.update(i.features)
        for i in self.features_data:#不用写.keys(), 只要value的话用.value()，全都要用.item(), val=mydict.get(key,<default value>),get不会在python中用到，因为会隐藏bug，没有key的时候，返回default value
            self.features_label[i] = sum(self.features_data[i].values())

    def label_count(self, label: str) -> int:
        if self.label_counts[label] is None:
            return 0
        return self.label_counts[label]

    def total_labels(self) -> int:
        return self.count

    def conditional_feature_count(self, label: str, feature: str) -> int:
        if self.features_data[label] is None:
            return 0
        # c = Counter(self.features_data.get(label))
        if self.features_data[label][feature] is None:
            return 0
        return self.features_data[label][feature]

    def labels(self) -> list[str]:
        data = list(self.label_counts.keys())
        return data

    def feature_vocab_size(self) -> int:
        return len(self.set_features)

    def total_feature_count_for_class(self, label: str) -> int:
        return self.features_label[label]


class NaiveBayesClassifier:
    def __init__(self, k: float):
        self.k: float = k
        self.counter: InstanceCounter = InstanceCounter()

    def train(self, instances: Iterable[ClassificationInstance]) -> None:
        self.counter.count_instances(instances)

    def classify(self, features: list[str]) -> str:
        a = list()
        record = self.counter.labels()
        for data in record:
            temp = self.log_posterior_prob(features,data)
            curr = (data, temp)
            a.append(curr)
        res = max(a, key=lambda x: x[1])
        return res[0]

    def prior_prob(self, label: str) -> float:
        a = float(self.counter.label_count(label))
        b = float(self.counter.total_labels())
        return float(a/b)

    def likelihood_prob(self, feature: str, label) -> float:
        if label not in self.counter.label_counts:
            return float(0)
        numerator = float(self.counter.conditional_feature_count(label,feature)+self.k)
        denominator = float(self.counter.total_feature_count_for_class(label)+self.k*self.counter.feature_vocab_size())
        if denominator == 0:
            return float(0)
        return float(numerator/denominator)

    def log_posterior_prob(self, features: list[str], label: str) -> float:
        a = float(self.prior_prob(label))
        sum_b = float(0.0)
        for fea in features:
            if fea in self.counter.features:
                sum_b += float(math.log(self.likelihood_prob(fea,label)))
        return float(math.log(a)+sum_b)

    def test(
        self, instances: Iterable[ClassificationInstance]
    ) -> tuple[list[str], list[str]]:
        a = list()
        b = list()
        for curr in instances:
            b.append(curr.label)
            a.append(self.classify(curr.features))
        res = (a,b)
        return res



# MODIFY THIS AND DO THE FOLLOWING:
# - Inherit from UnigramAirlineSentimentFeatureExtractor or BigramAirlineSentimentFeatureExtractor
#   (instead of object) to get an implementation for the extract_features method.
# - Change `self.k` below based on your tuning experiments.
class TunedAirlineSentimentFeatureExtractor(UnigramAirlineSentimentFeatureExtractor):
    def __init__(self) -> None:
        self.k = 0.5  # CHANGE ME
