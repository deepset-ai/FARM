from farm.utils import span_to_string

class Span:
    def __init__(self,
                 start,
                 end,
                 score=None,
                 sample_idx=None,
                 n_samples=None,
                 classification=None,
                 unit=None,
                 pred_str=None,
                 id=None,
                 level=None):
        self.start = start
        self.end = end
        self.score = score
        self.unit = unit
        self.sample_idx = sample_idx
        self.classification = classification
        self.n_samples = n_samples
        self.pred_str = pred_str
        self.id = id
        self.level = level

    def to_list(self):
        return [self.pred_str, self.start, self.end, self.score, self.sample_idx]

    def __str__(self):
        if self.pred_str is None:
            pred_str = "is_impossible"
        else:
            pred_str = self.pred_str
        ret = f"answer: {pred_str}\n" \
              f"score: {self.score}"
        return ret

    def __repr__(self):
        return str(self)

class DocumentPred:
    """ Contains a collection of Span predictions for one document. Used in Question Answering. Also contains all
    attributes needed to generate the appropriate output json"""
    def __init__(self,
                 id,
                 document_text,
                 question,
                 preds,
                 no_ans_gap,
                 token_offsets,
                 context_window_size):
        self.id = id
        self.preds = preds
        self.n_samples = preds[0].n_samples
        self.document_text = document_text
        self.question = question
        self.no_ans_gap = no_ans_gap
        self.token_offsets = token_offsets
        self.context_window_size = context_window_size

    def __str__(self):
        preds_str = "\n".join([f"{p}" for p in self.preds])
        ret = f"id: {self.id}\n" \
              f"document: {self.document_text}\n" \
              f"preds:\n{preds_str}"
        return ret

    def __repr__(self):
        return str(self)

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
                    "no_ans_gap": self.no_ans_gap # Add no_ans_gap to current no_ans_boost for switching top prediction
                }
            ],
        }
        return ret

    def answers_to_json(self):
        ret = []

        # iterate over the top_n predictions of the one document
        for span in self.preds:
            string = span.pred_str
            start_t = span.start
            end_t = span.end
            score = span.score

            _, ans_start_ch, ans_end_ch = span_to_string(start_t, end_t, self.token_offsets, self.document_text)
            context_string, context_start_ch, context_end_ch = self.create_context(ans_start_ch, ans_end_ch, self.document_text)
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
        preds = [x.to_list() for x in self.preds]
        ret = {"id": self.id,
               "preds": preds}
        return ret


