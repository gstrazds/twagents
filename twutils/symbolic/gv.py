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


