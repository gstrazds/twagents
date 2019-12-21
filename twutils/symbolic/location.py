# from typing import Optional
# from symbolic import gv
from .gv import logger
from .game import GameInstance
from .util import text_similarity
# from .event import NewEntityEvent, NewActionRecordEvent


class Location:
    """
    Each visited location contains information about entities, successful
    interactions, and connections to other locations.

    """
    def __init__(self, name='', description=''):
        if name:
            self._name = name
        else:
            self._name        = self.extract_name(description)
        self._description = description
        self._entities    = []
        self._action_records = {}   # action : ActionRec(p_valid, response)

    @property
    def name(self):
        return self._name

    @property
    def description(self):
        return self._description

    @description.setter
    def description(self, value):
        self._description = value

    @staticmethod
    def extract_name(description):
        """ Extracts the name of a location from a description. """
        stripped = description.strip()
        if "\n" in stripped:
            return stripped[:stripped.index("\n")]
        return stripped

    @property
    def entities(self):
        return self._entities

    def add_entity(self, entity) -> bool:
        if not self.has_entity_with_name(entity.name):
            # gv.event_stream.push(NewEntityEvent(entity))
            self._entities.append(entity)
            return True
        return False

    def get_entity_by_name(self, entity_name):
        """
        Returns an entity with the given name if it exists at this
        location or None if no such entity exists.

        """
        for entity in self.entities:
            if entity.has_name(entity_name):
                return entity
        return None

    def del_entity(self, entity):
        if entity in self._entities:
            self._entities.remove(entity)
        else:
            logger.warning("WARNING Location.del_entity could not find entity {}".format(entity.name))

    def has_entity(self, entity):
        return self.has_entity_with_name(entity.name) or \
            self.has_entity_with_description(entity.description)

    def has_entity_with_name(self, entity_name):
        return self.get_entity_by_name(entity_name) is not None

    def get_entity_by_description(self, entity_description):
        """
        Returns an entity with the given description if it exists at this
        location or None if no such entity exists.

        """
        for entity in self.entities:
            if text_similarity(entity.description, entity_description, substring_match=True) > 95:
                return entity
        return None

    def has_entity_with_description(self, entity_description):
        return self.get_entity_by_description(entity_description) is not None

    @property
    def action_records(self):
        return self._action_records

    # def add_action_record(self, action, p_valid, result_text) -> bool:
    #     """ Records an action, the probability it succeeded, and the text response. """
    #     if not isinstance(action, Action):
    #         raise ValueError("Expected Action. Got {}".format(type(action)))
    #     self._action_records[action] = (p_valid, result_text)
    #     return True
    #
    # def has_action_record(self, action):
    #     if not isinstance(action, Action):
    #         raise ValueError("Expected Action. Got {}".format(type(action)))
    #     return action in self._action_records
    #
    # def get_action_record(self, action):
    #     """ Returns (p_valid, result_text) for an action or None if it doesn't exist. """
    #     if not isinstance(action, Action):
    #         raise ValueError("Expected Action. Got {}".format(type(action)))
    #     return self.action_records[action] if self.has_action_record(action) else None

    def reset(self, gi: GameInstance):
        """ Reset to a state resembling start of game. """
        # Move all the entities back to their original locations
        to_remove = []
        for entity in self.entities:
            entity.reset()
            init_loc = entity._init_loc
            if init_loc is None or init_loc == self:
                continue
            init_loc.add_entity(entity)
            to_remove.append(entity)
        for entity in to_remove:
            self.entities.remove(entity)

    def to_string(self, prefix=''):
        s = prefix + "Loc<{}>:".format(self.name)
        for entity in self._entities:
            s += "\n" + entity.to_string(prefix + "  ")
        return s

    def __str__(self):
        return self.name

    def __repr__(self):
        return str(self)


class Inventory(Location):
    """
    Player inventory is represented as a location.

    """
    def __init__(self):
        super().__init__()
        self._name = 'Inventory'

    def __iter__(self):
        self._index = 0
        return self

    def __next__(self):
        if self._index == len(self.entities):
            raise StopIteration
        entity = self.entities[self._index]
        self._index += 1
        return entity

    def remove(self, entity):
        self._entities.remove(entity)

    def __str__(self):
        return "Inventory[" + str(', '.join([item.name for item in self.entities]))+"]"

    def __repr__(self):
        return str(self)


class UnknownLocation(Location):
    """
    Holder for entities that we know about, but don't (yet) know where they are.

    """
    def __init__(self):
        super().__init__()
        self._name = 'Unknown Location'

    def __iter__(self):
        self._index = 0
        return self

    def __next__(self):
        if self._index == len(self.entities):
            raise StopIteration
        entity = self.entities[self._index]
        self._index += 1
        return entity

    def remove(self, entity):
        self._entities.remove(entity)

    def __str__(self):
        return "<Unknown Location>[" + str(', '.join([item.name for item in self.entities]))+"]"

    def __repr__(self):
        return str(self)
