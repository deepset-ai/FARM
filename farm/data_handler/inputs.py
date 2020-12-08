from typing import List, Union


# class Question:
#     def __init__(self, text: str, uid: str=None):
#         self.text = text
#         self.uid = uid
#
#     def to_dict(self):
#         ret = {"question": self.text,
#                "id": self.uid,
#                "answers": []}
#         return ret
#
#
# class QAInput:
#     def __init__(self, doc_text: str, questions: Union[List[Question], Question]):
#         self.doc_text = doc_text
#         if type(questions) == Question:
#             self.questions = [questions]
#         else:
#             self.questions = questions
#
#     def to_dict(self):
#         questions = [q.to_dict() for q in self.questions]
#         ret = {"qas": questions,
#                "context": self.doc_text}
#         return ret

class QAPair:
    def __init__(self):
        doc_title
        doc_id
        doc_url
        question               # Or questions?
        source                 # squad / nq / squad+

    @classmethod
    def from_squad(cls):
        ...

    @classmethod
    def from_nq(cls):
        ...

    @classmethod
    def from_squad_plus(cls):
        ...

    @classmethod
    def from_inference_dict(cls):
        ...

    def save(self, save_file):
        ...

    def get_id(self):           # Different sources assign ids differently
        ...

    def to_clean_text(self):    # The format that we use to init Basket
        ...


class Question:
    """ This object by itself does not have reference to relevant document.
    My preference is to incorporate this in QAPair.
    By incorporating though, we might have copies of document when we have multi Q for one doc"""
    def __init__(self):
        text
        id
        answers

class Annotation:
    def __init__(self):
        id
        unit    #i.e. char or word
        answer_type
        text
        start
        end
        long answer?
        ...

    def to_char(self):
        ...

    def to_word(self):
        ...

    def fill_missing(self):     # calculate end ch index for Squad, or generate string for NQ
        ...


