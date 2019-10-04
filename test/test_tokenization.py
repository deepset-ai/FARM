import logging
from farm.modeling.tokenization import BertTokenizer, tokenize_with_metadata


def test_bert_a_never_split_chars(caplog):
    caplog.set_level(logging.CRITICAL)
    lang_model = "bert-base-cased"

    tokenizer = BertTokenizer.from_pretrained(
        pretrained_model_name_or_path=lang_model,
        do_lower_case=False,
        never_split_chars=["-","_","/"])

    tokenizer.add_custom_vocab("samples/tokenizer/custom_vocab.txt")
    #tokenizer.add_special_tokens()

    basic_text = "Some Text with neverseentokens plus !215?#. and a combined-token_with/chars"

    # original tokenizer from transformer repo
    tokenized = tokenizer.tokenize(basic_text)
    assert tokenized == ['Some', 'Text', 'with', 'neverseentokens', 'plus', '!', '215', '?', '#', '.', 'and', 'a', 'combined', '##-', '##tok', '##en', '##_', '##with', '##/', '##cha', '##rs']

    # ours with metadata
    tokenized_meta = tokenize_with_metadata(text=basic_text, tokenizer=tokenizer,max_seq_len=64)
    assert tokenized_meta["tokens"] == tokenized
    assert tokenized_meta["offsets"] ==  [0, 5, 10, 15, 31, 36, 37, 40, 41, 42, 44, 48, 50, 58, 59, 62, 64, 65, 69, 70, 73]
    assert tokenized_meta["start_of_word"] == [True, True, True, True, True, True, False, False, False, False, True, True, True, False, False, False, False, False, False, False, False]


def test_bert_tokenizer(caplog):
    caplog.set_level(logging.CRITICAL)

    lang_model = "bert-base-cased"

    #Note: we must actively set never_split_chars to None here, since it got set to other values before in other tests and seems to change the class definition somewhere in transformer repo
    tokenizer = BertTokenizer.from_pretrained(
        pretrained_model_name_or_path=lang_model,
        do_lower_case=False,
        never_split_chars=None)
    #tokenizer.add_special_tokens()

    basic_text = "Some Text with neverseentokens plus !215?#. and a combined-token_with/chars"

    # original tokenizer from transformer repo
    tokenized = tokenizer.tokenize(basic_text)
    assert tokenized == ['Some', 'Text', 'with', 'never', '##see', '##nto', '##ken', '##s', 'plus', '!', '215', '?', '#', '.', 'and', 'a', 'combined', '-', 'token', '_', 'with', '/', 'ch', '##ars']

    # ours with metadata
    tokenized_meta = tokenize_with_metadata(text=basic_text, tokenizer=tokenizer,max_seq_len=64)
    assert tokenized_meta["tokens"] == tokenized
    assert tokenized_meta["offsets"] == [0, 5, 10, 15, 20, 23, 26, 29, 31, 36, 37, 40, 41, 42, 44, 48, 50, 58, 59, 64, 65, 69, 70, 72]
    assert tokenized_meta["start_of_word"] == [True, True, True, True, False, False, False, False, True, True, False, False, False, False, True, True, True, False, False, False, False, False, False, False]


def test_bert_custom_vocab(caplog):
    caplog.set_level(logging.CRITICAL)

    lang_model = "bert-base-cased"

    #Note: we must actively set never_split_chars to None here, since it got set to other values before in other tests and seems to change the class definition somewhere in transformer repo
    tokenizer = BertTokenizer.from_pretrained(
        pretrained_model_name_or_path=lang_model,
        do_lower_case=False,
        never_split_chars=None)

    tokenizer.add_custom_vocab("samples/tokenizer/custom_vocab.txt")
    #tokenizer.add_special_tokens()

    basic_text = "Some Text with neverseentokens plus !215?#. and a combined-token_with/chars"

    # original tokenizer from transformer repo
    tokenized = tokenizer.tokenize(basic_text)
    assert tokenized == ['Some', 'Text', 'with', 'neverseentokens', 'plus', '!', '215', '?', '#', '.', 'and', 'a', 'combined', '-', 'token', '_', 'with', '/', 'ch', '##ars']

    # ours with metadata
    tokenized_meta = tokenize_with_metadata(text=basic_text, tokenizer=tokenizer,max_seq_len=64)
    assert tokenized_meta["tokens"] == tokenized
    assert tokenized_meta["offsets"] == [0, 5, 10, 15, 31, 36, 37, 40, 41, 42, 44, 48, 50, 58, 59, 64, 65, 69, 70, 72]
    assert tokenized_meta["start_of_word"] == [True, True, True, True, True, True, False, False, False, False, True, True, True, False, False, False, False, False, False, False]

if(__name__=="__main__"):
    test_bert_tokenizer()