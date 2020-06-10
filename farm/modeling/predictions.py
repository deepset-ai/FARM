from farm.utils import span_to_string
from abc import ABC, abstractclassmethod

class Pred(ABC):
    def __init__(self,
                 id,
                 prediction,
                 context):
        self.id = id
        self.prediction = prediction
        self.context = context

    def to_json(self):
        raise NotImplementedError


# class Span:
#     def __init__(self,
#                  start,
#                  end,
#                  score=None,
#                  sample_idx=None,
#                  n_samples=None,
#                  classification=None,
#                  unit=None,
#                  pred_str=None,
#                  id=None,
#                  level=None):
#         self.start = start
#         self.end = end
#         self.score = score
#         self.unit = unit
#         self.sample_idx = sample_idx
#         self.classification = classification
#         self.n_samples = n_samples
#         self.pred_str = pred_str
#         self.id = id
#         self.level = level
#
#     def to_list(self):
#         return [self.pred_str, self.start, self.end, self.score, self.sample_idx]
#
#     def __str__(self):
#         if self.pred_str is None:
#             pred_str = "is_impossible"
#         else:
#             pred_str = self.pred_str
#         ret = f"answer: {pred_str}\n" \
#               f"score: {self.score}"
#         return ret
#
#     def __repr__(self):
#         return str(self)

class QAAnswer:
    def __init__(self,
                 offset_answer_start,
                 offset_answer_end,
                 score,
                 answer_type,
                 offset_unit,
                 aggregation_level,
                 answer=None,
                 answer_support=None,
                 offset_answer_support_start=None,
                 offset_answer_support_end=None,
                 context=None,
                 offset_context_start=None,
                 offset_context_end=None,
                 probability=None,
                 sample_idx=None,
                 n_samples_in_doc=None,
                 document_id=None,
                 passage_id=None):
        # TODO THIS IS A LOT OF ATTRIBUTES - REMOVE SOME WITH METHODS
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
        self.offset_unit  = offset_unit
        self.aggregation_level = aggregation_level

        self.document_id = document_id
        self.passage_id = passage_id
        # TODO This seems to be necessary in order to align QA preds of len n_docs with Text Classification preds
        # TODO of len n_passages
        self.n_samples_in_doc = n_samples_in_doc

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


class QAPred(Pred):
    """Question Answering predictions for a passage or a document"""
    def __init__(self,
                 id,
                 prediction,
                 context,
                 question,
                 question_id,
                 token_offsets,         # TODO only needed for to_json() - can we get rid?
                 context_window_size,   # TODO Do we really need this?
                 aggregation_level,
                 answer_types=[],
                 ground_truth_answer=None,
                 no_answer_gap=None,

                 ):
        super().__init__(id,
                         prediction,
                         context)
        self.question = question
        self.question_id = question_id
        self.ground_truth_answer = ground_truth_answer
        self.no_answer_gap = no_answer_gap
        self.answer_types = answer_types
        self.n_samples = prediction[0].n_samples_in_doc

        self.token_offsets = token_offsets  # TODO only needed for to_json() - can we get rid?
        self.context_window_size = context_window_size  # TODO Do we really need this?
        self.aggregation_level = aggregation_level

        if len(self.prediction) > 0:
            assert type(self.prediction[0]) == QAAnswer

    def to_json(self):
        answers = self.answers_to_json()
        ret = {
            "task": "qa",
            "predictions": [
                {
                    "question": self.question,
                    "question_id": self.id,
                    "ground_truth": None,
                    "answers": answers,
                    "no_ans_gap": self.no_answer_gap # Add no_ans_gap to current no_ans_boost for switching top prediction
                }
            ],
        }
        return ret

    def answers_to_json(self):
        ret = []

        # iterate over the top_n predictions of the one document
        for qa_answer in self.prediction:
            string = qa_answer.answer
            start_t = qa_answer.offset_answer_start
            end_t = qa_answer.offset_answer_end
            score = qa_answer.score

            _, ans_start_ch, ans_end_ch = span_to_string(start_t, end_t, self.token_offsets, self.context)
            context_string, context_start_ch, context_end_ch = self.create_context(ans_start_ch, ans_end_ch, self.context)
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
        preds = [x.to_list() for x in self.prediction]
        ret = {"id": self.id,
               "preds": preds}
        return ret

