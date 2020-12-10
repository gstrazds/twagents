from collections import namedtuple
from symbolic import util
from .entity_state import EntityState
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
NONEXISTENCE_LOC = 'NOWHERE'
CONTAINED_LOCATION = 'C_LOC'
SUPPORTED_LOCATION = 'S_LOC'


ConnectionRelation = namedtuple('ConnectionRelation', 'from_location, direction')


def _conrel2str(conrel: ConnectionRelation):
    if not conrel:
        return "*"
    else:
        return f"{conrel.direction if conrel.direction else 'dirUNK'}:{conrel.from_location.name}"


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
        CONTAINED_LOCATION,
        SUPPORTED_LOCATION
    ]

    def __init__(self, name=None, description='', entitytype=None):
        assert name, "An Entity *must* have a name"
        self._names       = [name]  # List of names for the entity
        self._description = description
        self._type        = entitytype
        self._discovered = False
        self._entities = ()  # an empty tuple

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

    def is_a(self, entitytype):
        if self._type is None:
            return None
        return self._type == entitytype

    @property
    def description(self):
        return self._description

    @description.setter
    def description(self, value):
        self._description = value

    def reset(self, kg):   #kg: KnowledgeGraph
        pass

    @property
    def entities(self):
        return self._entities

    @property
    def parent(self):   # for building compositional hierarchy
        return None     # implemented by subclasses

class Location(Entity):
    """
    Each visited location contains information about entities, successful
    interactions, and connections to other locations.

    """
    def __init__(self, name='', description='', parent=None, entitytype=None):
        if not name:
            name = self.extract_name(description)
        if entitytype and entitytype != 'r' \
                and entitytype != UNKNOWN_LOCATION \
                and entitytype != NONEXISTENCE_LOC:
            assert parent is not None, f"only ROOMs should have no parent in the knowledge graph: {name}:{entitytype}"
        super().__init__(name=name, description=description, entitytype=entitytype)
        # if name:
        #     self._name = name
        # else:
        #    self._name = self.extract_name(description)
        #self._description = description
        self._action_records = {}  # action : ActionRec(p_valid, response)
        self._entities    = []
        self._parent      = parent   # link to a containing Entity (Location/Place or Thing )
        self._visit_count = 0

    @staticmethod
    def extract_name(description):
        """ Extracts the name of a location from a description. """
        stripped = description.strip()
        if "\n" in stripped:
            return stripped[:stripped.index("\n")]
        return stripped

    @staticmethod
    def is_unknown(location):  #NOTE: Location.is_unknown(None) => True
        return not isinstance(location, Location) or isinstance(location, UnknownLocation)

    # @property    # use inherited version from Entity
    # def is_known(self):
    #     self._discovered
    #     assert self._visit_count > 0 unless kg.groundtruth

    def visit(self):
        newly_discovered = self._visit_count == 0 or not self._discovered
        self._visit_count += 1
        self._discovered = True
        return newly_discovered

    @property
    def location(self):
        return self

    # @property
    # def entities(self):
    #     return self._entities

    def add_entity(self, entity) -> bool:
        if not self.has_entity_with_name(entity.name):
            self._entities.append(entity)
            if isinstance(entity, Door):
                if Location.is_unknown(entity.location):
                    entity.location = self
                elif Location.is_unknown(entity.location2):
                    entity.location2 = self
                elif Location.is_unknown(self):
                    print(f"SKIPPING MOVE OF DOOR '{entity}' to UnknownLocation")
                    pass
                else:
                    assert entity.location == self or entity.location2 == self,\
                        f"Can't move (to {self}) door:'{entity}' with 2 previous known locations {entity.location, entity.location2}"
            elif isinstance(entity, Thing):
                entity.location = self
            return True
        return False

    def get_entity_by_name(self, entity_name):
        """
        Returns an entity with the given name if it exists at this Location
        or None if no such entity exists.
        This is a shallow search: does not recurse to find entities contained or supported in/by entities.
        """
        for entity in self.entities:
            if entity.has_name(entity_name):
                return entity
        return None

    def del_entity(self, entity):
        if entity in self._entities:
            self._entities.remove(entity)
        else:
            print("***WARNING*** Location.del_entity could not find entity {}".format(entity.name))
            # logger.warning("WARNING Location.del_entity could not find entity {}".format(entity.name))

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
        for entity in list(self.entities):
            entity.reset(kg)
        #     init_loc = entity._init_loc
        #     if init_loc is None or init_loc == self:
        #         continue
        #     to_remove.append(entity)
        #     entity.location.del_entity(entity)
        #     init_loc.add_entity(entity)
        # for entity in to_remove:
        #     self.entities.remove(entity)

    @property
    def parent(self):
        return self._parent

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
    def __init__(self, owner: Entity = None):
        if owner:
            super().__init__(name=f"{owner.name}'s Inventory", entitytype=INVENTORY, parent=owner,
                             description=f"Inventory of items carried by {owner.name}")
        else:
            super().__init__(name='Inventory', entitytype=INVENTORY, description="Inventory of items carried by the Player")
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
        super().__init__(name='Unknown Location', entitytype=UNKNOWN_LOCATION)  # NOTE: parent=None)

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

    def __init__(self, name=None, description='', entitytype=None):  #, location=None):
        super().__init__(name=name, description=description, entitytype=entitytype)
        # self._names       = [name]  # List of names for the entity
        # self._description = description
        self._action_records = {} # verb : (p_valid, result_text)
        self._state       = EntityState()
        self._attributes  = []
        self._init_loc    = None
        self._current_loc = None   # location where this entity can currently be found
        # self._entities    = []
        self._container   = None   # if not None, a location holding objects contained by this entity
        self._supporting_surface = None   # if not None, a location with objects supported by/on this entity
        self._type        = entitytype
        # if location is not None:
        #     self.location = location

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

    def convert_to_container(self):
        if not self.is_container:
            self._container = Location(name=f"in_{self.name}", parent=self, entitytype=CONTAINED_LOCATION)
        return self._container

    @property
    def is_support(self) -> bool:
        return self._supporting_surface is not None

    def convert_to_support(self):
        if not self.is_support:
            self._supporting_surface = Location(name=f"on_{self.name}", parent=self, entitytype=SUPPORTED_LOCATION)
        return self._supporting_surface

    def add_entity(self, entity, kg, rel=None) -> bool:
        if rel == 'on':
            assert self.is_support is True
            return self._supporting_surface.add_entity(entity)
        elif rel == 'in':
            assert self.is_container is True
            return self._container.add_entity(entity)
        # elif rel == 'at':
        else:
            assert False, f"Unknown relation for Entity.add_entity({entity},rel={rel})"
        # self._entities.append(entity)
        return False


    def holds_entity(self, entity) -> bool:
        return self._container and self._container.has_entity(entity)

    def supports_entity(self, entity) -> bool:
        return self._supporting_surface and self._supporting_surface.has_entity(entity)

    def has_entity(self, entity):
        return self.holds_entity(entity) or self.supports_entity(entity)

    def del_entity(self, entity):
        if self.holds_entity(entity):
            return self._container.del_entity(entity)
        elif self.supports_entity(entity):
            return self._supporting_surface.del_entity(entity)
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
            if self._current_loc:
                self._current_loc.del_entity(self)  # can be in only one location at a time
            self._current_loc = new_location
            if not Location.is_unknown(new_location):
                if Location.is_unknown(self._init_loc):
                    # print(f"\tSETTING initial_location for {self} to: {new_location}")
                    self._init_loc = new_location

    @property
    def entities(self):  # return a list of controlled (contained and/or supported) entities)
        if self.is_container or self.is_support:
            controlled_entities = []
            if self._container:
                controlled_entities.extend(self._container.entities)
            if self._supporting_surface:
                controlled_entities.extend(self._supporting_surface.entities)
            return controlled_entities
        else:
            assert not self._entities  # default to an empty tuple
            return self._entities

    @property
    def parent(self):
        # if hasattr(self, "_parent"):   # in we have composite objects
        #     return self._parent
        return self._current_loc

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
            self._container.reset(kg)
        if self._supporting_surface:
            self._supporting_surface.reset(kg)
        if not Location.is_unknown(self._init_loc):
            self.location = self._init_loc
        self.state.reset(self)

    def open(self) -> bool:
        if self._type == DOOR or self.is_container:
            self.state.open()
        if self.is_container:
            self._container.visit()
        else:
            print(f"WARNING: attempting to open non-container: {self}")
            return False
        return self.state.is_open

    def close(self) -> bool:
        if self._type == DOOR or self.is_container:
            self.state.close()
        else:  #if not self.is_container:
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
        if self._supporting_surface:
            s += ('\n' + prefix + self._supporting_surface.to_string(prefix + "  "))
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
    def __init__(self, name=None, description=None):  #, location=None):
        super().__init__(name=name, description=description, entitytype=DOOR)  #, location=location)
        self._2nd_loc = None   # location (room) to which the door leads
        self._init_loc2 = None
        self.direction_from_loc1 = None
        self.direction_from_loc2 = None

    @property
    def location(self):
        return super().location

    @property
    def location2(self):
        return self._2nd_loc

    @location.setter
    def location(self, new_location: Location):
        if new_location != self._current_loc:
            prev_location = self._current_loc
            if prev_location and prev_location is not self.location2:  # special check for door in 2x UnknownLocation
                if not Location.is_unknown(prev_location):
                    assert False, \
                        f"Door ({self}) should not move from one location ({prev_location}) to another ({new_location})"
                prev_location.del_entity(self)  # can only be in one location at a time
            self._current_loc = new_location
            if not Location.is_unknown(new_location):
                if Location.is_unknown(self._init_loc):
                    # print(f"SETTING initial_location for {self} to: {new_location}")
                    self._init_loc = new_location
                if not self.direction_from_loc1:
                    self.direction_from_loc1 = ConnectionRelation(from_location=new_location, direction=None)

    @location2.setter
    def location2(self, new_location: Location):
        if new_location != self._2nd_loc:
            prev_location = self._2nd_loc
            if prev_location and prev_location is not self.location:  # special check for door in 2x UnknownLocation
                if not Location.is_unknown(prev_location):
                    assert False, \
                        f"Door ({self}) should not move from one location ({prev_location}) to another ({new_location})"
                prev_location.del_entity(self)
            self._2nd_loc = new_location
            if not Location.is_unknown(new_location):
                assert self.location != new_location, "Both sides of a door shouldn't be the same location"
                if Location.is_unknown(self._init_loc2):
                    # print(f"SETTING initial_location2 for {self} to: {new_location}")
                    self._init_loc2 = new_location
                if not self.direction_from_loc2:
                    self.direction_from_loc2 = ConnectionRelation(from_location=new_location, direction=None)

    def reset(self, kg):
        super().reset(kg)
        #NOTE: since doors can't be moved, we dont have to reset their locations
        # if not Location.is_unknown(self._init_loc2):
        #     self._2nd_loc = self._init_loc2

    def add_direction_rel(self, rel: ConnectionRelation):
        if rel.from_location == self.location:
            self.direction_from_loc1 = rel
        elif rel.from_location == self.location2:
            self.direction_from_loc2 = rel

    def __str__(self):
        strout = self.name
        if self.direction_from_loc1 or self.direction_from_loc2:
            strout += f"[{_conrel2str(self.direction_from_loc1)} {_conrel2str(self.direction_from_loc2)}]"
        return strout

class Person(Thing):
    def __init__(self, name='Somebody', description='An entity with volition'):  #, location=None):
        super().__init__(name=name, description=description, entitytype=PERSON)  #, location=location)
        self._container = Inventory(owner=self)

    @property
    def inventory(self):
        return self._container

