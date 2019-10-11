import logging
from farm.modeling.tokenization import BertTokenizer, RobertaTokenizer, XLNetTokenizer, tokenize_with_metadata
import re

# deprecated:
# def test_bert_a_never_split_chars(caplog):
#     caplog.set_level(logging.CRITICAL)
#     lang_model = "bert-base-cased"
#
#     tokenizer = BertTokenizer.from_pretrained(
#         pretrained_model_name_or_path=lang_model,
#         do_lower_case=False,
#         never_split_chars=["-","_","/"])
#
#     basic_text = "Some Text with neverseentokens plus !215?#. and a combined-token_with/chars"
#
#     # original tokenizer from transformer repo
#     tokenized = tokenizer.tokenize(basic_text)
#     assert tokenized == ['Some', 'Text', 'with', 'never', '##see', '##nto', '##ken', '##s', 'plus', '!', '215', '?', '#','.', 'and', 'a', 'combined', '##-', '##tok', '##en', '##_', '##with', '##/', '##cha', '##rs']
#     # ours with metadata
#     tokenized_meta = tokenize_with_metadata(text=basic_text, tokenizer=tokenizer,max_seq_len=64)
#     assert tokenized_meta["tokens"] == tokenized
#     assert tokenized_meta["offsets"] == [0, 5, 10, 15, 31, 36, 37, 40, 41, 42, 44, 48, 50, 58, 59, 62, 64, 65, 69, 70, 73]
#     assert tokenized_meta["start_of_word"] == [True, True, True, True, True, True, False, False, False, False, True, True, True, False, False, False, False, False, False, False, False]


def test_bert_tokenizer_all_meta(caplog):
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


def test_all_tokenizer_on_special_cases(caplog):
    caplog.set_level(logging.CRITICAL)

    #Note: we must actively set never_split_chars to None here, since it got set to other values before in other tests and seems to change the class definition somewhere in transformer repo
    bert_tokenizer = BertTokenizer.from_pretrained(
        pretrained_model_name_or_path="bert-base-cased",
        do_lower_case=False,
        never_split_chars=None)

    roberta_tokenizer = RobertaTokenizer.from_pretrained(
        pretrained_model_name_or_path="roberta-base")

    xlnet_tokenizer = XLNetTokenizer.from_pretrained(pretrained_model_name_or_path="xlnet-base-cased")

    tokenizers = [roberta_tokenizer, bert_tokenizer, xlnet_tokenizer]

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
            tokenized_meta = tokenize_with_metadata(text=text, tokenizer=tokenizer, max_seq_len=128)

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

    #Note: we must actively set never_split_chars to None here, since it got set to other values before in other tests and seems to change the class definition somewhere in transformer repo
    tokenizer = BertTokenizer.from_pretrained(
        pretrained_model_name_or_path=lang_model,
        do_lower_case=False,
        never_split_chars=None)

    #deprecated: tokenizer.add_custom_vocab("samples/tokenizer/custom_vocab.txt")
    tokenizer.add_tokens(new_tokens=["neverseentokens"])

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
    test_all_tokenizer_on_special_cases()