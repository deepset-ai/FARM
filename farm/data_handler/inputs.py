from typing import List, Union


class Question:
    def __init__(self, text: str, id: str=None):
        self.text = text
        self.id = id

    def to_dict(self):
        ret = {}
        return ret


class QAInput:
    def __init__(self, doc_text: str, questions: Union[List[Question], Question], doc_id: str=None):
        self.doc_text = doc_text
        if type(questions) == Question:
            self.questions = [questions]
        else:
            self.questions = questions
        self.doc_id = doc_id

    def to_dict(self):
        questions = [q.text for q in self.questions]
        ret = {"qas": questions,
               "context": self.doc_text}
        if self.doc_id:
            ret["doc_id"] = self.doc_id
        return ret

