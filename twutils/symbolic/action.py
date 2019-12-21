from abc import ABC, abstractmethod
from collections import namedtuple
from symbolic import gv   # global constants
from symbolic.game import GameInstance
from symbolic.entity import Entity
from symbolic.attribute import Attribute

ActionRec = namedtuple('ActionRecord', 'p_valid, result_text', defaults=(0.7, ''))

class Action(ABC):
    """
    An action contains the generation and effects of a given text
    string.

    """
    def __init__(self, verb):
        self.verb = verb

    @abstractmethod
    def text(self):
        """ Generate the text equivalent of a given action. """
        pass

    def validate(self, response_text):
        """
        Determine if an action has succeded based on the textual game's
        textual response. Returns p(valid), the probability of the
        action having succeded.

        """
        return None

    def apply(self, gi: GameInstance):
        """
        Apply the action to the knowledge_graph. The effects of applying
        depend on which action being applied.

        """
        pass

    def recognized(self, gi: GameInstance):
        """ Returns true if action doesn't contain unrecognized words. """
        if gi.includes_unrecognized_words(self.text()):
            return False
        return True

    def __str__(self):
        return self.text()

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return hash(self.text())

    def __eq__(self, other):
        return self.text() == other.text()


class StandaloneAction(Action):
    """ An action that doesn't require any entities. """
    def __init__(self, verb):
        super().__init__(verb)

    def text(self):
        return self.verb

class SingleAction(Action):
    """ An action of the form: verb entity. """
    def __init__(self, verb, entity):
        super().__init__(verb)
        if not isinstance(entity, Entity):
            raise ValueError("Expected entity object, got {}".format(type(entity)))
        self.entity = entity

    def text(self):
        return "{} {}".format(self.verb, self.entity.name)


class DoubleAction(Action):
    """ An action of the form: verb Entity1 preposition Entity2 """
    def __init__(self, verb, entity1, preposition, entity2):
        super().__init__(verb)
        if not isinstance(entity1, Entity):
            raise ValueError("Expected entity object, got {}".format(type(entity1)))
        if not isinstance(entity2, Entity):
            raise ValueError("Expected entity object, got {}".format(type(entity2)))
        self.entity1 = entity1
        self.prep = preposition
        self.entity2 = entity2

    def text(self):
        return "{} {} {} {}".format(self.verb, self.entity1.name,
                                    self.prep, self.entity2.name)


class NavAction(StandaloneAction):
    def __init__(self, verb):
        super().__init__(verb)

    def apply(self, gi: GameInstance):
        to_loc = gi.kg.connections.navigate(gi.kg.player_location, self)
        assert to_loc, "Error: Unknown connection"
        gi.kg.set_player_location(to_loc, gi)


class ExamineAction(Action):
    def __init__(self, entity_name:str):
        super().__init__("examine")
        self.entity_name = entity_name

    def text(self):
        return "{} {}".format(self.verb, self.entity_name)

    def apply(self, gi: GameInstance):
        entity = Entity(self.entity_name, location=gi.kg.player_location, ) #description=response)
        gi.entity_at_location(entity, gi.kg.player_location)


class TakeAction(SingleAction):
    def __init__(self, entity):
        super().__init__("take", entity)

    def apply(self, gi: GameInstance):
        player_loc = gi.kg.player_location
        if player_loc.has_entity(self.entity):
            player_loc.del_entity(self.entity)
        else:
            gv.logger.warning("WARNING Took non-present entity {}".format(self.entity.name))
        # gi.kg.inventory.add_entity(self.entity)
        gi.entity_at_location(self.entity, gi.kg.inventory)
        self.entity.add_attribute(Portable)
        # gi.add_entity_attribute(self.entity, Portable)

    def validate(self, response_text):
        if 'taken' in response_text.lower() or \
           'already' in response_text.lower():
            return 1.
        else:
            return 0.


class TakeAllAction(StandaloneAction):
    def __init__(self):
        super().__init__('take all')


class DropAction(SingleAction):
    def __init__(self, entity):
        super().__init__("drop", entity)

    def apply(self, gi: GameInstance):
        assert self.entity in gi.kg.inventory
        gi.kg.inventory.remove(self.entity)
        gi.entity_at_location(self.entity, gi.kg.player_location)  # gi.kg.player_location.add_entity(self.entity)
        self.entity.add_attribute(Portable)
        # gi.add_entity_attribute(self.entity, Portable)

    def validate(self, response_text):
        if 'dropped' in response_text.lower():
            return 1.
        else:
            return 0.


class OpenAction(SingleAction):
    def __init__(self, entity):
        super().__init__("open", entity)

    def apply(self, gi: GameInstance):
        self.entity.state.open()
        gi.add_entity_attribute(self.entity, Openable)


class CloseAction(SingleAction):
    def __init__(self, entity):
        super().__init__("close", entity)

    def apply(self, gi: GameInstance):
        self.entity.add_attribute(Openable)
        self.entity.state.close()
        # gi.add_entity_attribute(self.entity, Openable)


class LockAction(SingleAction):
    def __init__(self, entity):
        super().__init__("lock", entity)

    def apply(self, gi: GameInstance):
        self.entity.add_attribute(Lockable)
        self.entity.state.lock()
        # gi.add_entity_attribute(self.entity, Lockable)

class LockWithAction(DoubleAction):
    def __init__(self, entity1, entity2):
        super().__init__("lock", entity1, "with", entity2)

    def apply(self, gi: GameInstance):
        self.entity1.add_attribute(Lockable)
        self.entity1.state.lock()
        # gi.add_entity_attribute(self.entity1, Lockable)

class UnlockAction(SingleAction):
    def __init__(self, entity):
        super().__init__("unlock", entity)

    def apply(self, gi: GameInstance):
        self.entity.add_attribute(Lockable)
        self.entity.state.unlock()
        # gi.add_entity_attribute(self.entity, Lockable)

class UnlockWithAction(DoubleAction):
    def __init__(self, entity1, entity2):
        super().__init__("unlock", entity1, "with", entity2)

    def apply(self, gi: GameInstance):
        self.entity.add_attribute(Lockable)
        self.entity1.state.unlock()
        # gi.add_entity_attribute(self.entity1, Lockable)

class TurnOnAction(SingleAction):
    def __init__(self, entity):
        super().__init__("turn on", entity)

    def apply(self, gi: GameInstance):
        self.entity.add_attribute(Switchable)
        self.entity.state.turn_on()
        #gi.add_entity_attribute(self.entity, Switchable)


class TurnOffAction(SingleAction):
    def __init__(self, entity):
        super().__init__("turn off", entity)

    def apply(self, gi: GameInstance):
        self.entity.add_attribute(Switchable)
        self.entity.state.turn_off()
        #gi.add_entity_attribute(self.entity, Switchable)


class ConsumeAction(SingleAction):
    """ An action that consumes the entity. """
    def __init__(self, verb, entity):
        super().__init__(verb, entity)

    def apply(self, gi: GameInstance):
        self.entity.add_attribute(Edible)
        self.entity.state.remove()
        # gi.add_entity_attribute(self.entity, Edible)


class MoveItemAction(DoubleAction):
    """ An action that moves an item. """
    def __init__(self, verb, entity1, prep, entity2):
        super().__init__(verb, entity1, prep, entity2)

    def apply(self, gi: GameInstance):
        # TODO: Should entity contain a reference to its own container?
        # move_entity(self.entity1, source_container, self.entity2, gi)
        pass


# Global Action Definitions
NoOp       = StandaloneAction('<NOOP>')  # special singleton -- explicitly ignored
DoNothing  = StandaloneAction('do nothing')
Look       = StandaloneAction('look')
Inventory  = StandaloneAction('inventory')
GoNorth      = NavAction('north')
GoSouth      = NavAction('south')
GoEast       = NavAction('east')
GoWest       = NavAction('west')
# NorthWest  = NavAction('northwest')
# SouthWest  = NavAction('southwest')
# NorthEast  = NavAction('northeast')
# SouthEast  = NavAction('southeast')
# Up         = NavAction('up')
# Down       = NavAction('down')
# Enter      = NavAction('enter')
# Exit       = NavAction('exit')
# Climb      = NavAction('climb')
# In         = NavAction('in')
# Out        = NavAction('out')
# GetUp      = StandaloneAction('get up')
TakeAll    = StandaloneAction('take all')
# Yes        = StandaloneAction('yes')
# No         = StandaloneAction('no')
Take       = lambda x: TakeAction(x)
Drop       = lambda x: DropAction(x)
Examine    = lambda x: ExamineAction(x)
Eat        = lambda x: ConsumeAction('eat', x)
Drink      = lambda x: ConsumeAction('drink', x)
# Swallow    = lambda x: ConsumeAction('swallow', x)
# Consume    = lambda x: ConsumeAction('consume', x)
Open       = lambda x: OpenAction(x)
Close      = lambda x: CloseAction(x)
Lock       = lambda x: LockAction(x)
Unlock     = lambda x: UnlockAction(x)
LockWith   = lambda x,y: LockWithAction(x)
UnlockWith = lambda x,y: UnlockWithAction(x)
TurnOn     = lambda x: TurnOnAction(x)
TurnOff    = lambda x: TurnOffAction(x)
# Light      = lambda x: TurnOnAction(x)
# Extinguish = lambda x: TurnOffAction(x)
# Move       = lambda x: SingleAction('move', x)
# Push       = lambda x: SingleAction('push', x)
# Pull       = lambda x: SingleAction('pull', x)
# Drag       = lambda x: SingleAction('drag', x)
# Lift       = lambda x: SingleAction('lift', x)
# GiveTo     = lambda x,y: MoveItemAction('give', x, 'to', y)
#PutIn      = lambda x,y: MoveItemAction('put', x, 'in', y)
PutIn      = lambda x,y: MoveItemAction('insert', x, 'into', y)
PutOn      = lambda x,y: MoveItemAction('put', x, 'on', y)
TakeFrom   = lambda x,y: MoveItemAction('take', x, 'from', y)
Search     = lambda x: SingleAction('search', x) # TODO: Create informative action
# Ask        = lambda x: SingleAction('ask', x)
# Talk       = lambda x: SingleAction('talk to', x)
# SayTo      = lambda x,y: DoubleAction('say', x, 'to', y)
# Kiss       = lambda x: SingleAction('kiss', x)
# Bribe      = lambda x: SingleAction('bribe', x)
# BuyFrom    = lambda x,y: MoveItemAction('buy', x, 'from', y)
# Attack     = lambda x: SingleAction('attack', x)
# AttackWith = lambda x,y: DoubleAction('attack', x, 'with', y)
# Kill       = lambda x: SingleAction('kill', x)
# KillWith   = lambda x,y: DoubleAction('kill', x, 'with', y)

SliceWith   = lambda x,y: DoubleAction('slice', x, 'with', y)
ChopWith    = lambda x,y: DoubleAction('chop', x, 'with', y)
DiceWith    = lambda x,y: DoubleAction('dice', x, 'with', y)
CookWith    = lambda x,y: DoubleAction('cook', x, 'with', y)
Prepare     = lambda x: SingleAction('prepare', x)  # x='meal'
PrepareMeal = StandaloneAction('prepare meal')
EatMeal = StandaloneAction('eat meal')

# Global Entity Attributes
Portable   = Attribute('portable',   [Take, Drop, TakeFrom, PutOn, PutIn])  # GiveTo,
Edible     = Attribute('edible',     [Eat, Drink])  # Swallow, Consume])
Drinkable  = Attribute('drinkable',     [Drink])  # Swallow, Consume])
# Moveable   = Attribute('moveable',   [Move, Push, Pull, Drag, Lift])
Switchable = Attribute('switchable', [TurnOn, TurnOff])
# Flammable  = Attribute('flammable',  [Light, Extinguish])
Openable   = Attribute('openable',   [Open, Close])
Lockable   = Attribute('lockable',   [Lock, Unlock, LockWith, UnlockWith])
# TODO: An Openable object may be a container. We should have logic to check for containment
Container  = Attribute('container',  [PutIn, TakeFrom])  #, Search])
Support    = Attribute('support',    [PutOn, TakeFrom])  #, Search])
# Person     = Attribute('person',     [Ask, Talk, SayTo, Kiss, Bribe, GiveTo, BuyFrom])
# Enemy      = Attribute('enemy',      [Attack, AttackWith, Kill, KillWith])

Cutable    = Attribute('cutable',   [SliceWith, ChopWith, DiceWith])
Sharp      = Attribute('cut_2',   [SliceWith, ChopWith, DiceWith])
Cookable   = Attribute('cookable',   [CookWith])
Cooker     = Attribute('cook_2',   [CookWith])
Preparable = Attribute('preparable', [Prepare])

