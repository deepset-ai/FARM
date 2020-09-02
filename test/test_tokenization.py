import logging
import pytest
import re
from transformers import BertTokenizer, BertTokenizerFast, RobertaTokenizer, XLNetTokenizer, RobertaTokenizerFast
from transformers import ElectraTokenizerFast

from farm.modeling.tokenization import Tokenizer, tokenize_with_metadata, truncate_sequences


TEXTS = [
    "This is a sentence",
    "Der entscheidende Pass",
    "This      is a sentence with multiple spaces",
    "力加勝北区ᴵᴺᵀᵃছজটডণত",
    "Thiso text is included tolod makelio sure Unicodeel is handled properly:",
    "This is a sentence...",
    "Let's see all on this text and. !23# neverseenwordspossible",
    """This is a sentence.
    With linebreak""",
    """Sentence with multiple


    newlines
    """,
    "and another one\n\n\nwithout space",
    "This is a sentence	with tab",
    "This is a sentence			with multiple tabs",
]


def test_basic_loading(caplog):
    caplog.set_level(logging.CRITICAL)
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

            assert len(trunc_a) + len(trunc_b) + tokenizer.num_special_tokens_to_add(pair=True) == max_seq_len


@pytest.mark.parametrize("model_name", ["bert-base-german-cased",
                         "google/electra-small-discriminator",
                         ])
def test_fast_tokenizer_with_examples(caplog, model_name):
    fast_tokenizer = Tokenizer.load(model_name, lower_case=False, use_fast=True)
    tokenizer = Tokenizer.load(model_name, lower_case=False, use_fast=False)

    for text in TEXTS:
            # plain tokenize function
            tokenized = tokenizer.tokenize(text)
            fast_tokenized = fast_tokenizer.tokenize(text)

            assert tokenized == fast_tokenized


@pytest.mark.parametrize("model_name", ["bert-base-german-cased",
                         "google/electra-small-discriminator",
                         ])
def test_fast_tokenizer_with_metadata_with_examples(caplog, model_name):
    fast_tokenizer = Tokenizer.load(model_name, lower_case=False, use_fast=True)
    tokenizer = Tokenizer.load(model_name, lower_case=False, use_fast=False)

    for text in TEXTS:
            # our tokenizer with metadata on "whitespace tokenized words"
            tokenized_meta = tokenize_with_metadata(text=text, tokenizer=tokenizer)
            fast_tokenized_meta = tokenize_with_metadata(text=text, tokenizer=fast_tokenizer)

            # verify that tokenization on full sequence is the same as the one on "whitespace tokenized words"
            assert tokenized_meta == fast_tokenized_meta, f"Failed using {tokenizer.__class__.__name__}"


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
    """Sentence with multiple


    newlines
    """,
    "and another one\n\n\nwithout space",
    "This is a sentence	with tab",
    "This is a sentence			with multiple tabs",
    ]

    for tokenizer in tokenizers:
        for text in texts:
            # Important: we don't assume to preserve whitespaces after tokenization.
            # This means: \t, \n " " etc will all resolve to a single " ".
            # This doesn't make a difference for BERT + XLNet but it does for roBERTa

            # 1. original tokenize function from transformer repo on full sentence
            standardized_whitespace_text = ' '.join(text.split()) # remove multiple whitespaces
            tokenized = tokenizer.tokenize(standardized_whitespace_text)

            # 2. our tokenizer with metadata on "whitespace tokenized words"
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


def test_fast_bert_custom_vocab(caplog):
    caplog.set_level(logging.CRITICAL)

    lang_model = "bert-base-cased"

    tokenizer = Tokenizer.load(
        pretrained_model_name_or_path=lang_model,
        do_lower_case=False, use_fast=True
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


@pytest.mark.parametrize("model_name, tokenizer_type", [
                         ("bert-base-german-cased", BertTokenizerFast),
                         ("google/electra-small-discriminator", ElectraTokenizerFast),
                         ])
def test_fast_tokenizer_type(caplog, model_name, tokenizer_type):
    caplog.set_level(logging.CRITICAL)

    tokenizer = Tokenizer.load(model_name, use_fast=True)
    assert type(tokenizer) is tokenizer_type


def test_fast_bert_tokenizer_strip_accents(caplog):
    caplog.set_level(logging.CRITICAL)

    tokenizer = Tokenizer.load("dbmdz/bert-base-german-uncased",
                               use_fast=True,
                               strip_accents=False)
    assert type(tokenizer) is BertTokenizerFast
    assert tokenizer._tokenizer._parameters['strip_accents'] is False
    assert tokenizer._tokenizer._parameters['lowercase']


def test_fast_electra_tokenizer(caplog):
    caplog.set_level(logging.CRITICAL)

    tokenizer = Tokenizer.load("dbmdz/electra-base-german-europeana-cased-discriminator",
                               use_fast=True)
    assert type(tokenizer) is ElectraTokenizerFast


@pytest.mark.parametrize("model_name", ["bert-base-cased", "distilbert-base-uncased", "deepset/electra-base-squad2"])
def test_detokenization_in_fast_tokenizers(model_name):
    tokenizer = Tokenizer.load(
        pretrained_model_name_or_path=model_name,
        use_fast=True
    )
    for text in TEXTS:
        tokens_with_metadata = tokenize_with_metadata(text, tokenizer)
        tokens = tokens_with_metadata["tokens"]

        detokenized = " ".join(tokens)
        detokenized = re.sub(r"(^|\s+)(##)", "", detokenized)

        detokenized_ids = tokenizer(detokenized, add_special_tokens=False)["input_ids"]
        detokenized_tokens = [tokenizer.decode([tok_id]).strip() for tok_id in detokenized_ids]

        assert tokens == detokenized_tokens


if __name__ == "__main__":
    test_all_tokenizer_on_special_cases()