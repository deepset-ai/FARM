####################################
###### JSON (REST API) FORMAT ######
####################################

# INPUT

input = [{"questions": ["What is X?"], "text":  "Some context containing the answer"}]

# OUTPUT

output= {
    "task": "qa",
    "predictions": [
        {
            "question": question,
            "question_id": id,
            "ground_truth": None,
            "answers": answers,
            "no_ans_gap": no_ans_gap # Add no_ans_gap to current no_ans_boost for switching top prediction
        }
    ],
}

answer =   {"score": score,
              "probability": -1,
              "answer": string,
              "offset_answer_start": ans_start_ch,
              "offset_answer_end": ans_end_ch,
              "context": context_string,
              "offset_context_start": context_start_ch,
              "offset_context_end": context_end_ch,
              "document_id": document_id}


###############################
###### SQUAD EVAL FORMAT ######
###############################

# INPUT

input = [{"qas": ["What is X?"], "context":  "Some context containing the answer"}]

# OUTPUT

output = {"id": basket_id,
          "preds": [[pred_str, start_t, end_t, score, sample_idx], ...]}
