import sys
import random
from symbolic import knowledge_graph
from symbolic import event, action, attribute
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


class GameInstance:
    def __init__(self, kg: knowledge_graph.KnowledgeGraph):
        self.event_stream = event.EventStream()
        self.kg = knowledge_graph.KnowledgeGraph()

    def action_at_location(self, action, location, p_valid, result_text):
        ev = location.add_action_record(action, p_valid, result_text)
        self.event_stream.push(ev)

    def entity_at_location(self, entity, location):
        ev = location.add_entity(entity)
        if ev:
            self.event_stream.push(ev)

    def entity_at_entity(self, entity1, entity2):
        ev = entity1.add_entity(entity2)
        if ev:
            self.event_stream.push(ev)

    def act_on_entity(self, action, entity, p_valid, result_text):
        ev = entity.add_action_record(action, p_valid, result_text)
        if ev:
            self.event_stream.push(ev)


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


# Global Action Definitions
DoNothing  = action.StandaloneAction('do nothing')
Look       = action.StandaloneAction('look')
Inventory  = action.StandaloneAction('inventory')
North      = action.NavAction('north')
South      = action.NavAction('south')
East       = action.NavAction('east')
West       = action.NavAction('west')
# NorthWest  = action.NavAction('northwest')
# SouthWest  = action.NavAction('southwest')
# NorthEast  = action.NavAction('northeast')
# SouthEast  = action.NavAction('southeast')
# Up         = action.NavAction('up')
# Down       = action.NavAction('down')
# Enter      = action.NavAction('enter')
# Exit       = action.NavAction('exit')
# Climb      = action.NavAction('climb')
# In         = action.NavAction('in')
# Out        = action.NavAction('out')
# GetUp      = action.StandaloneAction('get up')
TakeAll    = action.StandaloneAction('take all')
# Yes        = action.StandaloneAction('yes')
# No         = action.StandaloneAction('no')
Take       = lambda x: action.TakeAction(x)
Drop       = lambda x: action.DropAction(x)
Examine    = lambda x: action.ExamineAction(x)
Eat        = lambda x: action.ConsumeAction('eat', x)
Drink      = lambda x: action.ConsumeAction('drink', x)
# Swallow    = lambda x: action.ConsumeAction('swallow', x)
# Consume    = lambda x: action.ConsumeAction('consume', x)
Open       = lambda x: action.OpenAction(x)
Close      = lambda x: action.CloseAction(x)
Lock       = lambda x: action.LockAction(x)
Unlock     = lambda x: action.UnlockAction(x)
LockWith   = lambda x,y: action.LockWithAction(x)
UnlockWith = lambda x,y: action.UnlockWithAction(x)
TurnOn     = lambda x: action.TurnOnAction(x)
TurnOff    = lambda x: action.TurnOffAction(x)
# Light      = lambda x: action.TurnOnAction(x)
# Extinguish = lambda x: action.TurnOffAction(x)
# Move       = lambda x: action.SingleAction('move', x)
# Push       = lambda x: action.SingleAction('push', x)
# Pull       = lambda x: action.SingleAction('pull', x)
# Drag       = lambda x: action.SingleAction('drag', x)
# Lift       = lambda x: action.SingleAction('lift', x)
# GiveTo     = lambda x,y: action.MoveItemAction('give', x, 'to', y)
#PutIn      = lambda x,y: action.MoveItemAction('put', x, 'in', y)
PutIn      = lambda x,y: action.MoveItemAction('insert', x, 'into', y)
PutOn      = lambda x,y: action.MoveItemAction('put', x, 'on', y)
TakeFrom   = lambda x,y: action.MoveItemAction('take', x, 'from', y)
Search     = lambda x: action.SingleAction('search', x) # TODO: Create informative action
# Ask        = lambda x: action.SingleAction('ask', x)
# Talk       = lambda x: action.SingleAction('talk to', x)
# SayTo      = lambda x,y: action.DoubleAction('say', x, 'to', y)
# Kiss       = lambda x: action.SingleAction('kiss', x)
# Bribe      = lambda x: action.SingleAction('bribe', x)
# BuyFrom    = lambda x,y: action.MoveItemAction('buy', x, 'from', y)
# Attack     = lambda x: action.SingleAction('attack', x)
# AttackWith = lambda x,y: action.DoubleAction('attack', x, 'with', y)
# Kill       = lambda x: action.SingleAction('kill', x)
# KillWith   = lambda x,y: action.DoubleAction('kill', x, 'with', y)

SliceWith   = lambda x,y: action.DoubleAction('slice', x, 'with', y)
ChopWith    = lambda x,y: action.DoubleAction('chop', x, 'with', y)
DiceWith    = lambda x,y: action.DoubleAction('dice', x, 'with', y)
CookWith    = lambda x,y: action.DoubleAction('cook', x, 'with', y)
Prepare     = lambda x: action.SingleAction('prepare', x)  # x='meal'

# Global Entity Attributes
Portable   = attribute.Attribute('portable',   [Take, Drop, TakeFrom, PutOn, PutIn])  # GiveTo,
Edible     = attribute.Attribute('edible',     [Eat, Drink])  # Swallow, Consume])
# Moveable   = attribute.Attribute('moveable',   [Move, Push, Pull, Drag, Lift])
# Switchable = attribute.Attribute('switchable', [TurnOn, TurnOff])
# Flammable  = attribute.Attribute('flammable',  [Light, Extinguish])
Openable   = attribute.Attribute('openable',   [Open, Close])
Lockable   = attribute.Attribute('lockable',   [Lock, Unlock, LockWith, UnlockWith])
# TODO: An Openable object may be a container. We should have logic to check for containment
Container  = attribute.Attribute('container',  [PutIn, TakeFrom])  #, Search])
Support    = attribute.Attribute('support',    [PutOn, TakeFrom])  #, Search])
# Person     = attribute.Attribute('person',     [Ask, Talk, SayTo, Kiss, Bribe, GiveTo, BuyFrom])
# Enemy      = attribute.Attribute('enemy',      [Attack, AttackWith, Kill, KillWith])

Cutable    = attribute.Attribute('cutable',   [SliceWith, ChopWith, DiceWith])
Cookable   = attribute.Attribute('cookable',   [CookWith])

