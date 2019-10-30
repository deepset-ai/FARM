import logging
from farm.modeling.tokenization import Tokenizer, tokenize_with_metadata, truncate_sequences
from transformers import BertTokenizer, RobertaTokenizer, XLNetTokenizer
import re


def test_basic_loading(caplog):

    tokenizer = Tokenizer.load(
        pretrained_model_name_or_path="bert-base-cased",
        do_lower_case=True
        )
    assert type(tokenizer) == BertTokenizer
    assert tokenizer.basic_tokenizer.do_lower_case == True

    tokenizer = Tokenizer.load(
        pretrained_model_name_or_path="xlnet-base-cased",
        do_lower_case=True
        )
    assert type(tokenizer) == XLNetTokenizer
    assert tokenizer.do_lower_case == True

    tokenizer = Tokenizer.load(
        pretrained_model_name_or_path="roberta-base"
        )
    assert type(tokenizer) == RobertaTokenizer


def test_bert_tokenizer_all_meta(caplog):
    caplog.set_level(logging.CRITICAL)

    lang_model = "bert-base-cased"

    tokenizer = Tokenizer.load(
        pretrained_model_name_or_path=lang_model,
        do_lower_case=False
        )

    basic_text = "Some Text with neverseentokens plus !215?#. and a combined-token_with/chars"

    # original tokenizer from transformer repo
    tokenized = tokenizer.tokenize(basic_text)
    assert tokenized == ['Some', 'Text', 'with', 'never', '##see', '##nto', '##ken', '##s', 'plus', '!', '215', '?', '#', '.', 'and', 'a', 'combined', '-', 'token', '_', 'with', '/', 'ch', '##ars']

    # ours with metadata
    tokenized_meta = tokenize_with_metadata(text=basic_text, tokenizer=tokenizer)
    assert tokenized_meta["tokens"] == tokenized
    assert tokenized_meta["offsets"] == [0, 5, 10, 15, 20, 23, 26, 29, 31, 36, 37, 40, 41, 42, 44, 48, 50, 58, 59, 64, 65, 69, 70, 72]
    assert tokenized_meta["start_of_word"] == [True, True, True, True, False, False, False, False, True, True, False, False, False, False, True, True, True, False, False, False, False, False, False, False]

def test_save_load(caplog):
    caplog.set_level(logging.CRITICAL)

    lang_names = ["bert-base-cased", "roberta-base", "xlnet-base-cased"]
    tokenizers = []
    for lang_name in lang_names:
        t = Tokenizer.load(lang_name, lower_case=False)
        t.add_tokens(new_tokens=["neverseentokens"])
        tokenizers.append(t)

    basic_text = "Some Text with neverseentokens plus !215?#. and a combined-token_with/chars"

    for tokenizer in tokenizers:
        save_dir = f"testsave"
        tokenizer_type = tokenizer.__class__.__name__
        tokenizer.save_pretrained(save_dir)
        tokenizer_loaded = Tokenizer.load(save_dir, tokenizer_class=tokenizer_type)
        tokenized_before = tokenize_with_metadata(text=basic_text, tokenizer=tokenizer)
        tokenized_after = tokenize_with_metadata(text=basic_text, tokenizer=tokenizer_loaded)
        assert tokenized_before == tokenized_after

def test_truncate_sequences(caplog):
    caplog.set_level(logging.CRITICAL)

    lang_names = ["bert-base-cased", "roberta-base", "xlnet-base-cased"]
    tokenizers = []
    for lang_name in lang_names:
        t = Tokenizer.load(lang_name, lower_case=False)
        tokenizers.append(t)

    # artificial sequences (could be tokens, offsets, or anything else)
    seq_a = list(range(10))
    seq_b = list(range(15))
    max_seq_len = 20
    for tokenizer in tokenizers:
        for strategy in ["longest_first", "only_first","only_second"]:
            trunc_a, trunc_b, overflow = truncate_sequences(seq_a=seq_a,seq_b=seq_b,tokenizer=tokenizer,
                                                        max_seq_len=max_seq_len, truncation_strategy=strategy)

            assert len(trunc_a) + len(trunc_b) + tokenizer.num_added_tokens(pair=True) == max_seq_len


def test_all_tokenizer_on_special_cases(caplog):
    caplog.set_level(logging.CRITICAL)

    lang_names = ["bert-base-cased", "roberta-base", "xlnet-base-cased"]
    tokenizers = []
    for lang_name in lang_names:
        t = Tokenizer.load(lang_name, lower_case=False)
        tokenizers.append(t)

    texts = [
     "This is a sentence",
     "Der entscheidende Pass",
    "This      is a sentence with multiple spaces",
    "力加勝北区ᴵᴺᵀᵃছজটডণত",
     "Thiso text is included tolod makelio sure Unicodeel is handled properly:",
   "This is a sentence...",
   "Let's see all on this text and. !23# neverseenwordspossible",
    """This is a sentence.
    With linebreak""",
    "This is a sentence with    tab"]

    for tokenizer in tokenizers:
        for text in texts:
            # Important: we don't assume to preserve whitespaces after tokenization.
            # This means: \t, \n " " etc will all resolve to a single " ".
            # This doesn't make a difference for BERT + XLNet but it does for roBERTa

            # 1. original tokenize function from transformer repo on full sentence
            standardized_whitespace_text = ' '.join(text.split()) # remove multiple whitespaces

            tokenized = tokenizer.tokenize(standardized_whitespace_text)
            tokenized_by_word = []
            # 2. original tokenize function from transformer repo on "whitespace tokenized words"
            for i, tok in enumerate(text.split(" ")):
                if i == 0:
                    tokenized_tok = tokenizer.tokenize(tok)
                else:
                    try:
                        tokenized_tok = tokenizer.tokenize(tok, add_prefix_space=True)
                    except TypeError:
                        tokenized_tok = tokenizer.tokenize(tok)
                tokenized_by_word.extend(tokenized_tok)
            assert tokenized == tokenized_by_word

            # 3. our tokenizer with metadata on "whitespace tokenized words"
            tokenized_meta = tokenize_with_metadata(text=text, tokenizer=tokenizer)

            # verify that tokenization on full sequence is the same as the one on "whitespace tokenized words"
            assert tokenized_meta["tokens"] == tokenized, f"Failed using {tokenizer.__class__.__name__}"

            # verify that offsets align back to original text
            if text == "力加勝北区ᴵᴺᵀᵃছজটডণত":
                # contains [UNK] that are impossible to match back to original text space
                continue
            for tok, offset in zip(tokenized_meta["tokens"], tokenized_meta["offsets"]):
                #subword-tokens have special chars depending on model type. In order to align with original text we need to get rid of them
                tok = re.sub(r"^(##|Ġ|▁)", "", tok)
                #tok = tokenizer.decode(tokenizer.convert_tokens_to_ids(tok))
                original_tok = text[offset:offset+len(tok)]
                assert tok == original_tok, f"Offset alignment wrong for {tokenizer.__class__.__name__} and text '{text}'"


def test_bert_custom_vocab(caplog):
    caplog.set_level(logging.CRITICAL)

    lang_model = "bert-base-cased"

    tokenizer = Tokenizer.load(
        pretrained_model_name_or_path=lang_model,
        do_lower_case=False
        )

    #deprecated: tokenizer.add_custom_vocab("samples/tokenizer/custom_vocab.txt")
    tokenizer.add_tokens(new_tokens=["neverseentokens"])

    basic_text = "Some Text with neverseentokens plus !215?#. and a combined-token_with/chars"

    # original tokenizer from transformer repo
    tokenized = tokenizer.tokenize(basic_text)
    assert tokenized == ['Some', 'Text', 'with', 'neverseentokens', 'plus', '!', '215', '?', '#', '.', 'and', 'a', 'combined', '-', 'token', '_', 'with', '/', 'ch', '##ars']

    # ours with metadata
    tokenized_meta = tokenize_with_metadata(text=basic_text, tokenizer=tokenizer)
    assert tokenized_meta["tokens"] == tokenized
    assert tokenized_meta["offsets"] == [0, 5, 10, 15, 31, 36, 37, 40, 41, 42, 44, 48, 50, 58, 59, 64, 65, 69, 70, 72]
    assert tokenized_meta["start_of_word"] == [True, True, True, True, True, True, False, False, False, False, True, True, True, False, False, False, False, False, False, False]



if(__name__=="__main__"):
    test_all_tokenizer_on_special_cases()