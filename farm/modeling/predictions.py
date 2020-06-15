from farm.utils import span_to_string
from abc import ABC
from typing import List, Optional, Any
from pydantic import BaseModel

class Pred(BaseModel):
    """
    Base class for predictions of every task. Note that it inherits from pydantic.BaseModel which creates an
    __init__() with the attributes defined in this class (i.e. id, prediction, context)
    """
    id: str
    prediction: List[Any]
    context: str


    def to_json(self):
        raise NotImplementedError

class QACandidate(BaseModel):
    """
    A single QA candidate answer. Note that it inherits from pydantic.BaseModel which builds the __init__() method.
    See class definition to find list of compulsory and optional arguments and also comments on how they are used.
    """

    # self.answer_type can be "is_impossible", "yes", "no" or "span"
    answer_type: str
    score: float
    probability: Optional[float] = None

    # If self.answer_type is "span", self.answer is a string answer span
    # Otherwise, it is None
    answer: Optional[str] = None
    offset_answer_start: int
    offset_answer_end: int

    # If self.answer_type is in ["yes", "no"] then self.answer_support is a text string
    # If self.answer is a string answer span or self.answer_type is "is_impossible", answer_support is None
    # TODO sample_idx can probably be removed since we have passage_id
    answer_support: Optional[str] = None
    offset_answer_support_start: Optional[int] = None
    offset_answer_support_end: Optional[int] = None
    sample_idx: Optional[int] = None

    # self.context is the document or passage where the answer is found
    context: Optional[str] = None
    offset_context_start: Optional[int] = None
    offset_context_end: Optional[int] = None

    # Offset unit is either "token" or "char"
    # Aggregation level is either "doc" or "passage"
    offset_unit: str
    aggregation_level: str

    n_samples_in_doc: Optional[int] = None
    document_id: Optional[str] = None
    passage_id: Optional[str] = None


    def to_doc_level(self, start, end):
        self.offset_answer_start = start
        self.offset_answer_end = end
        self.aggregation_level = "document"

    def add_answer(self, string):
        if string == "":
            self.answer = "is_impossible"
            assert self.offset_answer_end == -1
            assert self.offset_answer_start == -1
        else:
            self.answer = string
            assert self.offset_answer_end >= 0
            assert self.offset_answer_start >= 0

    def to_list(self):
        return [self.answer, self.offset_answer_start, self.offset_answer_end, self.score, self.sample_idx]


class QAPred(Pred):
    """Question Answering predictions for a passage or a document. The self.prediction attribute is populated by a
    list of QACandidate objects. Note that this object inherits from the Pred class which is why some of
    the attributes are found in the Pred class and not here. Pred in turn inherits from pydantic.BaseModel
    which creates an __init__() method. See class definition for required and optional arguments.
    """

    question: str
    token_offsets: List[int]
    context_window_size: int #TODO only needed for to_json() - can we get rid context_window_size, TODO Do we really need this?
    aggregation_level: str
    question_id: Optional[str]
    answer_types: Optional[List[str]] = []
    ground_truth_answer: Optional[str] = None
    no_answer_gap: Optional[float] = None
    n_samples: int = None

    def to_json(self, squad=False):
        answers = self.answers_to_json(squad)
        ret = {
            "task": "qa",
            "predictions": [
                {
                    "question": self.question,
                    "question_id": self.question_id,
                    "ground_truth": None,
                    "answers": answers,
                    "no_ans_gap": self.no_answer_gap # Add no_ans_gap to current no_ans_boost for switching top prediction
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
            score = qa_answer.score

            _, ans_start_ch, ans_end_ch = span_to_string(start_t, end_t, self.token_offsets, self.context)
            context_string, context_start_ch, context_end_ch = self.create_context(ans_start_ch, ans_end_ch, self.context)
            if squad:
                if string == "is_impossible":
                    string = ""
            curr = {"score": score,
                    "probability": -1,
                    "answer": string,
                    "offset_answer_start": ans_start_ch,
                    "offset_answer_end": ans_end_ch,
                    "context": context_string,
                    "offset_context_start": context_start_ch,
                    "offset_context_end": context_end_ch,
                    "document_id": self.id}
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
