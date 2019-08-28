import sys
import random
# from symbolic import attribute
import logging
import spacy

# Global RNG
rng = random.Random()

# Global logger
logger = logging.getLogger('nail')
logger.setLevel(logging.DEBUG)
dbg = logger.debug

# Global Event Stream
# event_stream = event.EventStream()

# Global Knowledge Graph
#kg = knowledge_graph.KnowledgeGraph()

# Actions that are dissallowed in any game
ILLEGAL_ACTIONS = ['restart', 'verbose', 'save', 'restore', 'score', 'quit', 'moves']


# Spacy NLP instance
# NOTE: to install spacy: need to do:
# pip install spacy
# python -m spacy download en_core_web_lg
try:
    nlp = spacy.load('en_core_web_lg')
except Exception as e:
    print("Failed to load \'en\' with exception {}. Try: python -m spacy download en_core_web_lg".format(e))
    sys.exit(1)

#### TextWorld: extras.command_templates:
#
#   'inventory',
#   'look',
#
#   'prepare meal',
#
#   'go east', 'go north', 'go south', 'go west',
#
#   'cook {f} with {oven}',
#   'cook {f} with {stove}',
#   'cook {f} with {toaster}',
#
#   'chop {f} with {o}',
#   'dice {f} with {o}',
#   'slice {f} with {o}',
#
#   'lock {c|d} with {k}',
#   'unlock {c|d} with {k}',
#
#   'close {c|d}',
#   'open {c|d}',
#
#   'take {o} from {c|s}',
#
#   'insert {o} into {c}',
#   'put {o} on {s}',
#
#   'drop {o}',
#   'take {o}',
#
#   'drink {f}',
#   'eat {f}',
#
#   'examine {o|t}',

COOK_WITH = {
    "grill": "BBQ",
    "bake": "oven",
    "roast": "oven",
    "fry": "stove",
    "toast": "toaster",
}

def convert_cooking_instruction(words, device:str):
    words.append("with")
    words.append(device)
    words[0] = "cook"  # convert the verb to generic "cook" (the specific verbs don't work as is in TextWorld)
    return " ".join(words)


def adapt_tw_instr(instr:str) -> str:
    if instr.startswith("chop ") or instr.startswith("dice ") or instr.startswith("slice "):
        return instr + " with the knife", ["knife"]
    words = instr.split()
    with_objs = []
    if words[0] in COOK_WITH:
        device = COOK_WITH[words[0]]
        with_objs.append(device)
        return convert_cooking_instruction(words, device), with_objs
    else:
        return instr, []

# Entity types for TextWorld CodaLab challenge #1 (recipe games)
OBJECT = 'o'
THING = 't'
SUPPORT = 's'
CONTAINER = 'c'
FOOD = 'f'
KEY = 'k'
PERSON = 'P'
ROOM = 'r'
DOOR = 'd'
OVEN = 'oven'
STOVE = 'stove'
TOASTER = 'toaster'
BBQ = 'bbq'
MEAL = 'meal'

RECIPE = 'RECIPE'
INGREDIENT = 'ingredient'
SLOT = 'slot'


