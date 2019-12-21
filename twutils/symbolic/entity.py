from symbolic.gv import OBJECT, THING,  PERSON, CONTAINER, SUPPORT, ROOM
from symbolic.gv import FOOD, DOOR, KEY, STOVE, OVEN, TOASTER, BBQ
from symbolic.gv import MEAL, RECIPE, INGREDIENT, SLOT
from symbolic import util
from symbolic.location import Location

class Entity:
    """
    An Entity represents an object or person encountered in a game.

    @args
    name: Name of the entity
    location: Location in which the entity was first encountered
    description: A long form description of the entity

    """
    entity_types = [
        OBJECT, THING,  PERSON, CONTAINER, SUPPORT, ROOM,
        FOOD, DOOR, KEY, STOVE, OVEN, TOASTER, BBQ,
        MEAL, RECIPE, INGREDIENT,
        # NORTH, WEST, EAST, SOUTH,
        SLOT,
    ]

    def __init__(self, name, location=None, description='', type=None):
        self._names       = [name] # List of names for the entity
        self._description = description
        self._action_records = {} # verb : (p_valid, result_text)
        # self._entities    = []
        self._state       = EntityState()
        self._attributes  = []
        self._init_loc    = location
        self._current_loc = None   # location where this entity can currently be found
        self._contains    = None   # if not None, a location for objects contained by this entity
        self._supports    = None   # if not None, a location with objects supported by/on this entity
        self._type        = type

    @property
    def name(self):
        return self._names[0]

    @name.setter
    def name(self, value):
        self._names[0] = value

    @property
    def names(self):
        return self._names

    def has_name(self, name):
        return name in self._names

    def add_name(self, new_name):
        if new_name in self.names:
            return
        if len(new_name.split(' ')) < len(self.name.split(' ')):
            self._names.insert(0, new_name)
        else:
            self._names.append(new_name)

    @property
    def description(self):
        return self._description

    @description.setter
    def description(self, value):
        self._description = value

    @property
    def action_records(self):
        return self._action_records

    def add_action_record(self, action, rec) -> bool:
        """
        Record an action that was applied to this object and the
        resulting game text.

        """
        was_new = action not in self._action_records
        self._action_records[action] = rec
        return was_new

    def has_action_record(self, action):
        return action in self._action_records

    def add_entity(self, entity, rel=None) -> bool:
        if rel == 'on':
            if self._supports is None:
                self._supports = Location(name=f"on_{self.name}")
            return self._supports.add_entity(entity)
        elif rel == 'in':
            if self._contains is None:
                self._contains = Location(name=f"in_{self.name}")
            return self._contains.add_entity(entity)
        # elif rel == 'at':
        else:
            assert False, f"Unknown relation for Entity.add_entity({entity},rel={rel})"
        # self._entities.append(entity)
        return False

    def contains_entity(self, entity) -> bool:
        return self._contains and self._contains.has_entity(entity)

    def supports_entity(self, entity) -> bool:
        return self._supports and self._supports.has_entity(entity)

    def del_entity(self, entity):
        if self.contains_entity(entity):
            return self._contains.del_entity(entity)
        elif self.supports_entity(entity):
            return self._supports.del_entity(entity)
        else:
            print(f"ENTITY_NOT_FOUND: {self.name}.del_entity({entity.name})")
        return False

    @property
    def attributes(self):
        return self._attributes

    def add_attribute(self, attribute) -> bool:
        if attribute not in self._attributes:
            self._attributes.append(attribute)
            return True
        return False

    def del_attribute(self, attribute):
        if attribute in self._attributes:
            self._attributes.remove(attribute)

    @property
    def state(self):
        return self._state

    def reset(self):
        """ Reset the entity to a state similar to when the game started. """
        # Remove all successful action records
        to_remove = []
        for action_record, (p_valid, result_text) in self.action_records.items():
            if p_valid > .5:
                to_remove.append(action_record)
        for action_record in to_remove:
            del self.action_records[action_record]

    def to_string(self, prefix=''):
        s = prefix + "Entity: {}".format(self.name)
        if self._action_records:
            for action, (p_valid, resp) in self._action_records.items():
                if p_valid > .5:
                    s += "\n  {}Action record: {} {} - {} (p={:.2f})".format(
                        prefix, action, self.name, util.clean(resp)[:80], p_valid)
        if self._contains:
            s += ('\n' + prefix + self._contains.to_string(prefix + "  "))
        if self._supports:
            s += ('\n' + prefix + self._supports.to_string(prefix + "  "))
        if self._attributes:
            s += "\n  " + prefix + "Attributes: "
            for attribute in self._attributes:
                s += attribute.to_string() + " "
        return s

    def __str__(self):
        return self.name

    def __repr__(self):
        return str(self)


class EntityState:
    """
    Keeps track of the current state of the entity.

    """
    def __init__(self):
        self.exists = True

    @property
    def openable(self):
        return hasattr(self, 'is_open')

    def open(self):
        self.is_open = True

    def close(self):
        self.is_open = False

    def lockable(self):
        return hasattr(self, 'is_locked')

    def lock(self):
        self.is_locked = True

    def unlock(self):
        self.is_locked = False

    def switchable(self):
        return hasattr(self, 'is_on')

    def turn_on(self):
        self.is_on = True

    def turn_off(self):
        self.is_on = False

    def remove(self):
        self.exists = False

    @property
    def cookable(self):
        return hasattr(self, 'is_cooked')

    def cook(self, cooked_state='cooked'):
        self.is_cooked = cooked_state

    def not_cooked(self):
        self.is_cooked = ''  # equiv to False, but can also be tested with str.startswith()

    @property
    def cuttable(self):
        return hasattr(self, 'is_cut')

    def cut(self, cut_state='cut'):  # 'sliced', 'chopped', 'diced', etc...
        self.is_cut = cut_state

    def not_cut(self):
        self.is_cut = ''  # equiv to False, but can also be tested with str.startswith()

    def __str__(self):
        pass
