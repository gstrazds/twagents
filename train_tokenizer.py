import glob
# from typing import List, Dict, Optional, Any, Tuple
import tokenizers
from tokenizers import Tokenizer
from tokenizers.pre_tokenizers import Whitespace   # WhitespaceSplit, Punctuation
from tokenizers import normalizers
from tokenizers.normalizers import Lowercase, Strip  #, Replace, Sequence

MAX_VOCAB_SIZE = 1024    # 500

def build_tokenizer(dir_list, max_vocab=None):
    if max_vocab is None:
        max_vocab = MAX_VOCAB_SIZE

    if not dir_list:
        PTHRU_DIR = ''
        GATA_PTHRU_DIR = ''
        dir_list = [
            PTHRU_DIR + "valid/*.pthru",
            PTHRU_DIR + "test/*.pthru",
            PTHRU_DIR + "train/*.pthru",
            GATA_PTHRU_DIR + "gata_valid/*.pthru",
            GATA_PTHRU_DIR + "gata_test/*.pthru",
            GATA_PTHRU_DIR + "gata_100/*.pthru",
            ]
    special_tokens = ['<PAD>', '<UNK>', '<S>', '</S>', '<bos>', '<eos>', '<NONE>', '<sep>', '<|>']

    # '+open', '+closed', '+roasted', '+baked', '+fried', '+raw',
    # '+sliced', '+diced', '+chopped', '++Carrying:', ]

    normalizer = normalizers.Sequence([Strip(), Lowercase()])
    pre_tokenizer = Whitespace()

    model = tokenizers.models.WordLevel(unk_token='<UNK>')
    # model = tokenizers.models.WordPiece()
    tokenizer = Tokenizer(model=model)

    tokenizer.add_special_tokens(special_tokens)
    tokenizer.add_tokens(['pathtrace'])
    # tokenizer.add_tokens([ COMMAND_TOKEN,])
    # tokenizer.add_tokens(['frosted-glass'])
    # tokenizer.add_tokens(['[UNK]'])  #'++Carrying:', 'hello', '+how', 'today'])
    tokenizer.normalizer = normalizer
    tokenizer.pre_tokenizer = pre_tokenizer

    filelist = []
    for dir_path in dir_list:
        print(dir_path)
        filelist.extend(glob.glob(dir_path))
    print(len(filelist))

    # token_strs = [tok for (tok, span) in pre_tokenizer.pre_tokenize_str(str1)]
    # print(token_strs)
    # unigram_trainer = tokenizers.trainers.UnigramTrainer()
    # trainer = tokenizers.trainers.WordPieceTrainer(vocab_size=max_vocab)
    trainer = tokenizers.trainers.WordLevelTrainer(vocab_size=max_vocab, special_tokens=special_tokens)

    tokenizer.train(files=filelist, trainer=trainer)
    return tokenizer


def save_tokenizer_to_json(tokenizer, filepath):
    # filepath = 'combined_tokenizer_new.json'
    vocab = tokenizer.get_vocab(with_added_tokens=True)
    print(len(vocab))
    slist = [(vocab[k], k) for k in vocab.keys()]
    for i, (tokid, tok) in enumerate(sorted(slist)):
        print(tokid, tok)
    # tokenizer.save('ftwc_tokenizer_new.json', pretty=True)
    tokenizer.save(filepath, pretty=True)


if __name__ == "__main__":
    import argparse
    import os
    parser = argparse.ArgumentParser(description="Train and save a tokenizer from a set of playthrough data files")

    parser.add_argument("--which",
                        help="Which set of games? (subdir of env.TWDATA_DIR): ftwc or gata [by default will use both]")
    parser.add_argument("--pthru-dirname",
                        help="Subdirectory name where the playthrough files can be found, defaults to 'playthru_data'")
    parser.add_argument("--tokenizer-input-globs", type=list, nargs="+",
                        help="(Advanced/optional) One or more patterns for playthrough data files")
    parser.add_argument("--tokenizer-filepath", default="tokenizer_new.json",
                        help="Output file path to use when saving the trained tokenizer")

    args = parser.parse_args()

    TWDATA_DIR = os.getenv('TWDATA_DIR', '/work2/gstrazds/twdata')
    if args.tokenizer_input_globs:
        glob_list = args.tokenizer_input_globs
    else:
        subdir_name = args.pthru_dirname if args.pthru_dirname else 'playthru_data'
        glob_list = []
        if args.which == 'ftwc' or args.which == 'both' or not args.which:
            PTHRU_DIR = f"{TWDATA_DIR}/ftwc/{subdir_name}/"
            glob_list += [
                PTHRU_DIR + "valid/*.pthru",
                PTHRU_DIR + "test/*.pthru",
                PTHRU_DIR + "train/*.pthru"
                ]
        if args.which == 'gata' or args.which == 'both' or not args.which:
            GATA_PTHRU_DIR = f"{TWDATA_DIR}/gata/{subdir_name}/"
            glob_list += [
                GATA_PTHRU_DIR + "gata_valid/*.pthru",
                GATA_PTHRU_DIR + "gata_test/*.pthru",
                GATA_PTHRU_DIR + "gata_train/*.pthru",
            ]

    tokenizer = build_tokenizer(glob_list)
    save_tokenizer_to_json(tokenizer, args.tokenizer_filepath)

