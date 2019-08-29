import torch
from farm.data_handler.processor import TextClassificationProcessor
from farm.infer import Inferencer
from farm.modeling.adaptive_model import AdaptiveModel
from farm.modeling.language_model import Bert
from farm.modeling.tokenization import BertTokenizer
from farm.utils import set_all_seeds, MLFlowLogger, initialize_device_settings
##########################
########## Settings
##########################
set_all_seeds(seed=42)
batch_size = 32
use_gpu = False
device, n_gpu = initialize_device_settings(use_cuda=use_gpu)
# 5. Extract embeddings with model in inference mode
# basic_texts = [
#     {"text": "HPV not closed [SEP] MEL applied"},
#     {"text": "james wordham: #date 16:40 utc   on initial climb air eng 4 bleed fault hpv not closed reported by: access staff no: 129635 [SEP] #date 16:40] pfr shows f/c 3621w040 and 3611fexc per tsm 361100810813a number 4 eng hp bleed vlv requires replacement. eng 4 hp vlv locked closed per mel 361105b (m)(o) to hold. (note on deactivating valve: valve confirmed not closed and notchy)authorisation number: qe772780"},
#     {"text": "#name #name: #date 03:58 utc   on climb: ecam air eng3 bleed fault: hp vlv not closed reported by: #name staff no: 323069 ref t00f20ap for part requirements. [SEP] [#date 03:58] defect confirmed in pfr: fc 3611fb5c: nil ecam present on arrival: tsm reqs hp vlv replacement: due nil gnd time mel 361105b [o] [m] applied and vlv deactivated in closed positionauthorisation number: qe524693"},
#     {"text": "serqovht #name 4b loc050 [SEP] t/s found precooler (7150ha4) leakage"},
#     {"text": "#name sambell: #date 00:15 utc   after engine shutdown ecam air eng 3 bleed reported by: #name staff no: 570093 [SEP] [#date 00:15] ebas test and ebas pressure sensor drift test c/out iaw amm 361100with nil faults #3 engine run c/out iaw amm 710000 confirmed correct operationauthorisation number: qe431122"},
#
# ]
basic_texts = [
    {"text": "HPV not closed [SEP] MEL applied"},
    {"text": "james wordham: #date 16:40 utc   on initial climb air eng 4 bleed fault hpv not closed reported by: access staff no: 129635 [SEP] #date 16:40] pfr shows f/c 3621w040 and 3611fexc per tsm 361100810813a number 4 eng hp bleed vlv requires replacement. eng 4 hp vlv locked closed per mel 361105b (m)(o) to hold. (note on deactivating valve: valve confirmed not closed and notchy)authorisation number: qe772780"},
    {"text": "#name #name: #date 03:58 utc   on climb: ecam air eng3 bleed fault: hp vlv not closed reported by: #name staff no: 323069 ref t00f20ap for part requirements. [SEP] [#date 03:58] defect confirmed in pfr: fc 3611fb5c: nil ecam present on arrival: tsm reqs hp vlv replacement: due nil gnd time mel 361105b [o] [m] applied and vlv deactivated in closed positionauthorisation number: qe524693"},
    {"text": "serqovht #name 4b loc050 [SEP] t/s found precooler (7150ha4) leakage"},
    {"text": "#name sambell: #date 00:15 utc   after engine shutdown ecam air eng 3 bleed reported by: #name staff no: 570093 [SEP] [#date 00:15] ebas test and ebas pressure sensor drift test c/out iaw amm 361100with nil faults #3 engine run c/out iaw amm 710000 confirmed correct operationauthorisation number: qe431122"},
    {"text": "408-408 49vu 228-228 [SEP] t/s found precooler (7150ha4) leakage"},
]
model = Inferencer.load("../../airbus-etlbmining/data/bert/models/tlb-fine-tuned/airbert-2.0.1", gpu=use_gpu, embedder_only=True)
result = model.extract_vectors(dicts=basic_texts, extraction_strategy="reduce_mean", extraction_layer=-1)
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
mat = np.array([r["vec"]for r in result])
similarities = cosine_similarity(mat)
print(similarities)