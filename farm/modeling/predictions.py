from abc import ABC
from typing import List, Any
import logging

from farm.utils import span_to_string

logger = logging.getLogger(__name__)

class Pred(ABC):
    """
    Base Abstract Class for predictions of every task.
    """

    def __init__(self,
                 id: str,
                 prediction: List[Any],
                 context: str):
        self.id = id
        self.prediction = prediction
        self.context = context

    def to_json(self):
        raise NotImplementedError


class QACandidate:
    """
    A single QA candidate answer.
    See class definition to find list of compulsory and optional arguments and also comments on how they are used.
    """

    def __init__(self,
                 answer_type: str,
                 score: str,
                 offset_answer_start: int,
                 offset_answer_end: int,
                 offset_unit: str,
                 aggregation_level: str,
                 probability: float = None,
                 answer: str = None,
                 answer_support: str = None,
                 offset_answer_support_start: int = None,
                 offset_answer_support_end: int = None,
                 sample_idx: int = None,
                 context: str = None,
                 offset_context_start: int = None,
                 offset_context_end: int = None,
                 n_samples_in_doc: int = None,
                 document_id: str = None,
                 passage_id: str = None,
                 ):
        # self.answer_type can be "is_impossible", "yes", "no" or "span"
        self.answer_type = answer_type
        self.score = score
        self.probability = probability

        # If self.answer_type is "span", self.answer is a string answer span
        # Otherwise, it is None
        self.answer = answer
        self.offset_answer_start = offset_answer_start
        self.offset_answer_end = offset_answer_end

        # If self.answer_type is in ["yes", "no"] then self.answer_support is a text string
        # If self.answer is a string answer span or self.answer_type is "is_impossible", answer_support is None
        # TODO sample_idx can probably be removed since we have passage_id
        self.answer_support = answer_support
        self.offset_answer_support_start = offset_answer_support_start
        self.offset_answer_support_end = offset_answer_support_end
        self.sample_idx = sample_idx

        # self.context is the document or passage where the answer is found
        self.context = context
        self.offset_context_start = offset_context_start
        self.offset_context_end = offset_context_end

        # Offset unit is either "token" or "char"
        # Aggregation level is either "doc" or "passage"
        self.offset_unit = offset_unit
        self.aggregation_level = aggregation_level

        self.n_samples_in_doc = n_samples_in_doc
        self.document_id = document_id
        self.passage_id = passage_id

    def add_cls(self, predicted_class: str):
        """
        Adjust the final QA prediction depending on the prediction of the classification head (e.g. for binary answers in NQ)
        Currently designed so that the QA head's prediction will always be preferred over the Classification head

        :param predicted_class: the predicted class value
        :return: None
        """

        if predicted_class in ["yes", "no"] and self.answer != "no_answer":
            self.answer_support = self.answer
            self.answer = predicted_class
            self.answer_type = predicted_class
            self.offset_answer_support_start = self.offset_answer_start
            self.offset_answer_support_end = self.offset_answer_end

    def to_doc_level(self, start, end):
        self.offset_answer_start = start
        self.offset_answer_end = end
        self.aggregation_level = "document"

    def add_answer(self, string):
        if string == "":
            self.answer = "is_impossible"
            if self.offset_answer_start != -1 or self.offset_answer_end != -1:
                logger.error(f"Something went wrong in tokenization. We have start and end offsets: "
                             f"{self.offset_answer_start, self.offset_answer_end} with an empty answer. "
                             f"\nContext: {self.context}")
        else:
            self.answer = string
            if self.offset_answer_start == -1 or self.offset_answer_end == -1:
                logger.error(f"Something went wrong in tokenization. We have start and end offsets: "
                             f"{self.offset_answer_start, self.offset_answer_end} with answer: {string}. "
                             f"\nContext: {self.context}")

    def to_list(self):
        return [self.answer, self.offset_answer_start, self.offset_answer_end, self.score, self.sample_idx]


class QAPred(Pred):
    """Question Answering predictions for a passage or a document. The self.prediction attribute is populated by a
    list of QACandidate objects. Note that this object inherits from the Pred class which is why some of
    the attributes are found in the Pred class and not here.
    See class definition for required and optional arguments.
    """

    def __init__(self,
                 id: str,
                 prediction: List[QACandidate],
                 context: str,
                 question: str,
                 token_offsets: List[int],
                 context_window_size: int,
                 aggregation_level: str,
                 answer_types: List[str] = None,
                 ground_truth_answer: str = None,
                 no_answer_gap: float = None,
                 n_samples: int = None,
                 question_id: int = None,
                 ):
        super().__init__(id, prediction, context)
        self.question = question
        self.token_offsets = token_offsets
        self.context_window_size = context_window_size  # TODO only needed for to_json() - can we get rid context_window_size, TODO Do we really need this?
        self.aggregation_level = aggregation_level
        self.answer_types = answer_types
        self.ground_truth_answer = ground_truth_answer
        self.no_answer_gap = no_answer_gap
        self.n_samples = n_samples
        self.question_id = question_id

    def to_json(self, squad=False):
        answers = self.answers_to_json(squad)
        ret = {
            "task": "qa",
            "predictions": [
                {
                    "question": self.question,
                    "question_id": self.question_id,
                    "ground_truth": self.ground_truth_answer,
                    "answers": answers,
                    "no_ans_gap": self.no_answer_gap, # Add no_ans_gap to current no_ans_boost for switching top prediction
                }
            ],
        }
        return ret

    def answers_to_json(self, squad=False):
        ret = []

        # iterate over the top_n predictions of the one document
        for qa_answer in self.prediction:
            string = qa_answer.answer
            start_t = qa_answer.offset_answer_start
            end_t = qa_answer.offset_answer_end

            _, ans_start_ch, ans_end_ch = span_to_string(start_t, end_t, self.token_offsets, self.context)
            context_string, context_start_ch, context_end_ch = self.create_context(ans_start_ch,
                                                                                   ans_end_ch,
                                                                                   self.context)
            if squad:
                if string == "is_impossible":
                    string = ""
            curr = {"score": qa_answer.score,
                    "probability": None,
                    "answer": string,
                    "offset_answer_start": ans_start_ch,
                    "offset_answer_end": ans_end_ch,
                    "context": context_string,
                    "offset_context_start": context_start_ch,
                    "offset_context_end": context_end_ch,
                    "document_id": qa_answer.document_id}
            ret.append(curr)
        return ret

    def create_context(self, ans_start_ch, ans_end_ch, clear_text):
        if ans_start_ch == 0 and ans_end_ch == 0:
            return "", 0, 0
        else:
            len_text = len(clear_text)
            midpoint = int((ans_end_ch - ans_start_ch) / 2) + ans_start_ch
            half_window = int(self.context_window_size / 2)
            context_start_ch = midpoint - half_window
            context_end_ch = midpoint + half_window
            # if we have part of the context window overlapping start or end of the passage,
            # we'll trim it and use the additional chars on the other side of the answer
            overhang_start = max(0, -context_start_ch)
            overhang_end = max(0, context_end_ch - len_text)
            context_start_ch -= overhang_end
            context_start_ch = max(0, context_start_ch)
            context_end_ch += overhang_start
            context_end_ch = min(len_text, context_end_ch)
        context_string = clear_text[context_start_ch: context_end_ch]
        return context_string, context_start_ch, context_end_ch

    def to_squad_eval(self):
        return self.to_json(squad=True)
