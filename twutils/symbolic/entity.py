from symbolic import util
# from symbolic.location import Location
from .entity_state import EntityState
from .gv import logger
from .util import text_similarity

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
INVENTORY='I'

RECIPE = 'RECIPE'
INGREDIENT = 'ingredient'
SLOT = 'slot'

# define some additional special type codes
UNKNOWN_OBJ_TYPE = 'UNKNOWN'
UNKNOWN_LOCATION = 'UNK_LOC'
CONTAINED_LOCATION = 'C_LOC'


class Entity:
    """ Something that has a name (possibly several) and possibly a description.
    Locations are also entities."""

    entity_types = [
        # (type codes specific to the TextWorld platform. TODO: fix this!)
        OBJECT, THING,  PERSON, CONTAINER, SUPPORT, ROOM,
        FOOD, DOOR, KEY, STOVE, OVEN, TOASTER, BBQ,
        MEAL, RECIPE, INGREDIENT,
        # NORTH, WEST, EAST, SOUTH,
        SLOT,
        # and some non-TextWorld type codes
        UNKNOWN_OBJ_TYPE,
        UNKNOWN_LOCATION,
        CONTAINED_LOCATION
    ]

    def __init__(self, name=None, description='', type=None):
        assert name, "An Entity *must* have a name"
        self._names       = [name]  # List of names for the entity
        self._description = description
        self._type        = type
        self._discovered = False

    @property
    def name(self):
        return self._names[0]

    @name.setter
    def name(self, value):
        self._names[0] = value

    @property
    def is_known(self):
        return self._discovered

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

    def reset(self, kg):   #kg: KnowledgeGraph
        pass


class Location(Entity):
    """
    Each visited location contains information about entities, successful
    interactions, and connections to other locations.

    """
    def __init__(self, name='', description='', parent=None, type=None):
        if not name:
            name = self.extract_name(description)
        super().__init__(name=name, description=description, type=type)
        # if name:
        #     self._name = name
        # else:
        #    self._name = self.extract_name(description)
        #self._description = description
        self._action_records = {}  # action : ActionRec(p_valid, response)
        self._entities    = []
        self._parent      = None   # link to a containing Entity (Location/Place or Thing )
        self._visit_count = 0

    # @property
    # def name(self):
    #     return self._name
    #
    # @property
    # def description(self):
    #     return self._description
    #
    # @description.setter
    # def description(self, value):
    #     self._description = value

    @staticmethod
    def extract_name(description):
        """ Extracts the name of a location from a description. """
        stripped = description.strip()
        if "\n" in stripped:
            return stripped[:stripped.index("\n")]
        return stripped

    @staticmethod
    def is_unknown(location):  #NOTE: Location.is_unknown(None) => True
        return not isinstance(location, Location) or not location.is_known

    # @property    # use inherited version from Entity
    # def is_known(self):
    #     self._discovered
    #     assert self._visit_count > 0 unless kg.groundtruth

    def visit(self):
        if self._visit_count == 0 or not self._discovered:
            print(f"visit() DISCOVERED {self}")
        self._visit_count += 1
        self._discovered = True

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

    def reset(self, kg):  # KnowledgeGraph):
        """ Reset to a state resembling start of game. """
        # Move all the entities back to their original locations
        to_remove = []
        for entity in self.entities:
            entity.reset(kg)
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
    def __init__(self, player:Entity):
        super().__init__(name='Inventory', type=INVENTORY, description="Inventory of items carried by Player")
        # self._name = 'Inventory'

    def __iter__(self):
        self._index = 0
        return self

    def __next__(self):
        if self._index == len(self.entities):
            raise StopIteration
        entity = self.entities[self._index]
        self._index += 1
        return entity

    def visit(self):
        assert False

    @property
    def is_known(self):
        return True

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
        super().__init__(name='Unknown Location', type=UNKNOWN_LOCATION)
        # self._name = 'Unknown Location'

    def visit(self):
        assert False, "Should never be able to visit() the (conceptual, abstract) UnknownLocation"

    @property
    def is_known(self):
        assert self._discovered is False
        return False

    def __str__(self):
        return "<Unknown Location>[" + str(', '.join([item.name for item in self.entities]))+"]"

    def __repr__(self):
        return str(self)


class Thing(Entity):
    """
    An Entity represents an object or person encountered in a game.

    @args
    name: Name of the entity
    location: Location in which the entity was first encountered
    description: A long form description of the entity

    """

    def __init__(self, name=None, description='', location=None, type=None):
        super().__init__(name=name, description=description, type=type)
        # self._names       = [name]  # List of names for the entity
        # self._description = description
        self._action_records = {} # verb : (p_valid, result_text)
        self._state       = EntityState()
        self._attributes  = []
        self._init_loc    = location
        self._current_loc = location   # location where this entity can currently be found
        # self._entities    = []
        self._container   = None   # if not None, a location holding objects contained by this entity
        self._supports    = None   # if not None, a location with objects supported by/on this entity
        self._type        = type

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

    @property
    def is_container(self) -> bool:
        return self._container is not None

    @is_container.setter
    def is_container(self, boolval: bool):
        if boolval:
            if not self.is_container:
                self._container = Location(name=f"in_{self.name}")
        elif not boolval and self.is_container:
            assert False, "Cannot convert a container into a non-container"
        return

    @property
    def is_support(self) -> bool:
        return self._supports is not None

    @is_support.setter
    def is_support(self, boolval: bool):
        if boolval:
            if not self.is_support:
                self._supports = Location(name=f"on_{self.name}")
        elif not boolval and self.is_support:
            assert False, "Cannot convert a supporting object to non-supporting"
        return

    def add_entity(self, entity, rel=None) -> bool:
        if rel == 'on':
            self.is_support = True
            return self._supports.add_entity(entity)
        elif rel == 'in':
            self.is_container = True
            return self._container.add_entity(entity)
        # elif rel == 'at':
        else:
            assert False, f"Unknown relation for Entity.add_entity({entity},rel={rel})"
        # self._entities.append(entity)
        return False


    def holds_entity(self, entity) -> bool:
        return self._container and self._container.has_entity(entity)

    def supports_entity(self, entity) -> bool:
        return self._supports and self._supports.has_entity(entity)

    def has_entity(self, entity):
        return self.holds_entity(entity) or self.supports_entity(entity)

    def del_entity(self, entity):
        if self.holds_entity(entity):
            return self._container.del_entity(entity)
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
    def location(self):
        return self._current_loc

    @location.setter
    def location(self, new_location: Location):
        if new_location != self._current_loc:
            self._current_loc = new_location
            if not Location.is_unknown(new_location):
                #new_location and new_location.is_known:
                # not isinstance(new_location, UnknownLocation):
                if Location.is_unknown(self._init_loc):
                    #not self._init_loc or not self._init_loc.is_known:
                    #isinstance(self._init_loc, UnknownLocation):
                    print(f"SETTING initial_location for {self} to: {new_location}")
                    self._init_loc = new_location

    @property
    def state(self):
        return self._state

    def reset(self, kg):
        """ Reset the entity to a state similar to when the game started. """
        # Remove all successful action records
        to_remove = []
        for action_record, (p_valid, result_text) in self.action_records.items():
            if p_valid > .5:
                to_remove.append(action_record)
        for action_record in to_remove:
            del self.action_records[action_record]
        if self._container:
            for e in self._container.entities:
                e.reset(kg)
        if self._supports:
            for e in self._supports.entities:
                e.reset(kg)
        self.location = self._init_loc
        self.state.reset()

    def open(self) -> bool:
        if self.type == DOOR or self._is_container:
            self.state.open()
        if self.is_container:
            self._container.visit()
        else:
            print(f"WARNING: attempting to open non-container: {self}")
            return False
        return self.state.is_open

    def close(self) -> bool:
        if self.type == DOOR or self._is_container:
            self.state.close()
        if not self.is_container:
            print(f"WARNING: attempting to close non-container: {self}")
            return False
        return self.state.is_open is False

    def to_string(self, prefix=''):
        s = prefix + "Entity: {}".format(self.name)
        if self._action_records:
            for action, (p_valid, resp) in self._action_records.items():
                if p_valid > .5:
                    s += "\n  {}Action record: {} {} - {} (p={:.2f})".format(
                        prefix, action, self.name, util.clean(resp)[:80], p_valid)
        if self._container:
            s += ('\n' + prefix + self._container.to_string(prefix + "  "))
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


class Door(Thing):
    def __init__(self, name=None, description=None, location=None):
        super().__init__(name=name, description=description, type=DOOR, location=location)
        self._2nd_loc = None   # location (room) to which the door leads
        self._init_loc2 = None

    @property
    def location2(self):
        return self._2nd_loc

    @location2.setter
    def location2(self, new_location: Location):
        if new_location != self._2nd_loc:
            self._2nd_loc = new_location
            if not Location.is_unknown(new_location):
                # new_location.is_known:
                # not isinstance(new_location, UnknownLocation):
                if Location.is_unknown(self._init_loc2):
                    #not self._init_loc2 or not self._init_loc2.is_known:
                    #isinstance(self._init_loc2, UnknownLocation):
                    print(f"SETTING initial_location2 for {self} to: {new_location}")
                    self._init_loc2 = new_location


class Person(Thing):
    def __init__(self, name='Player', description='The protagonist', location=None):
        super().__init__(name=name, description=description, type=PERSON, location=location)
        self._container = Inventory(self)

    @property
    def inventory(self):
        return self._container

