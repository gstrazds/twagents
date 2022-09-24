import re
from fuzzywuzzy import fuzz
# from symbolic import gv
# from symbolic import event
# from symbolic.game import GameInstance
# from symbolic.action import Action
# from symbolic.location import Location


# def first_sentence(text):
#     """ Extracts the first sentence from text. """
#     tokens = gv.nlp(text)
#     return next(tokens.sents).merge().text
#
#
# def tokenize(description):
#     """ Returns a list of tokens in a string. """
#     doc = gv.nlp(description)
#     return [word.lower_ for word in doc]


def clean(s):
    """ Clean a string for compact output. """
    return s.replace('\n', ' ').strip()


def text_similarity(str1: str, str2: str, substring_match=False):
    if substring_match:
        similarity = fuzz.partial_ratio(str1, str2)
    else:
        similarity = fuzz.ratio(str1, str2)
    return similarity


# This list covers the common paterns. However, some games like
# loose.z5 and lostpig.z8 write custom responses that aren't included.
REGEXPS = [
    ".*That's not a verb I recognise.*",
    ".*I don't know the word \"(\w+)\.?\".*",
    ".*You used the word \"(\w+)\" in a way that I don't understand.*",
    ".*This story doesn't know the word \"(\w+)\.?\".*",
    ".*This story doesn't recognize the word \"(\w+)\.?\".*",
    ".*The word \"(\w+)\" isn't in the vocabulary that you can use.*",
    ".*You don't need to use the word \"(\w+)\" to finish this story.*",
    ".*You don't need to use the word \"(\w+)\" to complete this story.*",
    ".*Sorry, but the word \"(\w+)\" is not in the vocabulary you can use.*",
    ".*Sorry, but this story doesn't recognize the word \"(\w+)\.?\".*",
]
COMPILED_UNRECOGNIZED_REGEXPS = [re.compile(regexp) for regexp in REGEXPS]


