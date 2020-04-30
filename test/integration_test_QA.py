from farm.infer import Inferencer
from farm.data_handler.utils import write_squad_predictions
from farm.utils import initialize_device_settings
from farm.modeling.tokenization import Tokenizer
from farm.data_handler.processor import SquadProcessor
from farm.data_handler.data_silo import DataSilo
from farm.eval import Evaluator
from farm.evaluation import squad_evaluation
from farm.modeling.adaptive_model import AdaptiveModel
from pathlib import Path
import numpy as np
from dotmap import DotMap
from time import time
import logging
import os

def test_evaluation():
    ##########################
    ########## Settings
    ##########################
    lang_model = "deepset/roberta-base-squad2"
    do_lower_case = True

    data_dir = Path("testsave/data/squad20")
    evaluation_filename = "dev-v2.0.json"

    device, n_gpu = initialize_device_settings(use_cuda=True)


    # loading models and evals
    model = AdaptiveModel.convert_from_transformers(lang_model, device=device, task_type="question_answering")
    model.prediction_heads[0].no_ans_boost = 0
    model.prediction_heads[0].n_best = 1

    tokenizer = Tokenizer.load(pretrained_model_name_or_path=lang_model,do_lower_case=do_lower_case)
    processor = SquadProcessor(
        tokenizer=tokenizer,
        max_seq_len=256,
        label_list= ["start_token", "end_token"],
        metric="squad",
        train_filename=None,
        dev_filename=None,
        dev_split=0,
        test_filename=evaluation_filename,
        data_dir=data_dir,
        doc_stride=128,
    )

    starttime = time()

    data_silo = DataSilo(processor=processor, batch_size=50)
    model.connect_heads_with_processor(data_silo.processor.tasks, require_labels=True)
    evaluator = Evaluator(data_loader=data_silo.get_data_loader("test"), tasks=data_silo.processor.tasks, device=device)

    # 1. Test FARM internal evaluation
    results = evaluator.eval(model)
    f1_score = results[0]["f1"]*100
    em_score = results[0]["EM"]*100
    tnrecall = results[0]["top_n_recall"]*100
    elapsed = time() - starttime
    print(elapsed)

    gold_f1 = 80.56469
    gold_EM = 75.73486
    gold_tnrecall = 82.39703
    gold_elapsed = 95.11 ## TBD
    np.testing.assert_allclose(f1_score, gold_f1, rtol=0.001, err_msg=f"FARM Eval changed for f1 score by: {f1_score-gold_f1}")
    np.testing.assert_allclose(em_score, gold_EM, rtol=0.001, err_msg=f"FARM Eval changed for EM by: {em_score-gold_EM}")
    np.testing.assert_allclose(tnrecall, gold_tnrecall, rtol=0.001, err_msg=f"FARM Eval changed for top 1 recall by: {em_score-gold_EM}")
    #np.testing.assert_allclose(elapsed, gold_elapsed, rtol=0.1, err_msg=f"FARM Eval speed changed significantly: {elapsed - gold_elapsed}")


    # 2. Test FARM predictions with outside eval script
    starttime = time()
    model = Inferencer(model=model, processor=processor, task_type="question_answering",batch_size=50, gpu=device.type=="cuda")
    filename = data_dir / evaluation_filename
    result = model.inference_from_file(file=filename)

    elapsed = time() - starttime

    os.makedirs("testsave", exist_ok=True)
    write_squad_predictions(
        predictions=result,
        predictions_filename=filename,
        out_filename="testsave/predictions.json"
    )
    script_params = {"data_file": filename,
              "pred_file": "testsave/predictions.json",
              "na_prob_thresh" : 1,
              "na_prob_file": False,
              "out_file": False}
    results_official = squad_evaluation.main(OPTS=DotMap(script_params))
    f1_score = results_official["f1"]
    em_score = results_official["exact"]

    gold_f1 = 80.03626
    gold_EM = 76.69502
    gold_elapsed = 76.035 ## TBD
    np.testing.assert_allclose(f1_score, gold_f1, rtol=0.001,
                               err_msg=f"Eval with official script changed for f1 score by: {f1_score - gold_f1}")
    np.testing.assert_allclose(em_score, gold_EM, rtol=0.001,
                               err_msg=f"Eval with official script changed for EM by: {em_score - gold_EM}")
    np.testing.assert_allclose(elapsed, gold_elapsed, rtol=0.1,
                               err_msg=f"Inference speed changed significantly: {elapsed - gold_elapsed}")


if __name__ == "__main__":
    logging.disable(logging.WARNING)
    test_evaluation()