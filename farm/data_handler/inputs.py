from typing import List

class Question:
    def __init__(self, q: str, id: str=None):
        self.q = q
        self.id = id

    def to_dict(self):
        ret = {}

class QAInput:
    def __init__(self, doc_text: str, questions: List[Question], doc_id: str=None):
        self.doc_text = doc_text
        self.questions = questions
        self.doc_id = doc_id

    def to_dict(self):
        questions = [q.to_dict() for q in self.questions]
        ret = {}

