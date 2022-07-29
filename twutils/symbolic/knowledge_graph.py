# import os, sys
# from itertools import chain
from typing import Tuple

from symbolic.event import *
from symbolic.action import *
from symbolic.entity import ConnectionRelation, Entity, Thing, Location, Person, UnknownLocation, Door
from symbolic.entity import DOOR, ROOM, PERSON, INVENTORY, CONTAINER, SUPPORT, CONTAINED_LOCATION, SUPPORTED_LOCATION, NONEXISTENCE_LOC
from twutils.twlogic import reverse_direction
from .mention_index import MentionIndex

DIRECTION_ACTIONS = {
        'north_of': GoNorth,
        'south_of': GoSouth,
        'east_of': GoEast,
        'west_of': GoWest}

LOCATION_RELATIONS = ['at', 'in', 'on']

def simple_direction(direction):
    if direction.endswith('_of'):
        return direction[:-3]
    return direction

def map_action_to_direction(action):
    for key, val in DIRECTION_ACTIONS.items():
        if val == action:
            return key
    return None


def get_attributes_for_type(twvartype:str):
    attrib_list = []
    if twvartype == 't' \
    or twvartype == 'P' \
    or twvartype == 'I' \
    or twvartype == 'r':
        pass
    elif twvartype == 'o':
        attrib_list.append(Portable)
    elif twvartype == 'f':
        attrib_list.append(Portable)
        attrib_list.append(Edible)
        attrib_list.append(Cutable)
        attrib_list.append(Cookable)
    elif twvartype == 'c':
        attrib_list.append(Container)
        attrib_list.append(Openable)
    elif twvartype == 's':
        attrib_list.append(Support)
    elif twvartype == 'k':
        attrib_list.append(Portable)
    elif twvartype == 'd':
        attrib_list.append(Openable)
        # attrib_list.append(Lockable)
    elif twvartype == 'oven':
        attrib_list.append(Container)
        attrib_list.append(Openable)
    elif twvartype == 'stove':
        attrib_list.append(Support)
    elif twvartype == 'bbq':
        attrib_list.append(Cooker)
    elif twvartype == 'toaster':
        attrib_list.append(Cooker)
    elif twvartype == 'meal':
        attrib_list.append(Preparable)
    elif twvartype == 'ingredient':
        pass
    elif twvartype == 'slot':
        pass
    elif twvartype == 'RECIPE':
        pass
    else:
        print("Warning -- get_attributes_for_type() unexpected variable type:", twvartype)
    return attrib_list


def get_attributes_for_predicate(predicate, entity2, groundtruth=False):
    attrib_list = []
    if predicate == 'sharp':
        attrib_list.append(Sharp)
    elif predicate == 'closed':
        attrib_list.append(Openable)
    elif predicate == 'open':
        attrib_list.append(Openable)
    elif predicate == 'locked':
        attrib_list.append(Lockable)
    elif predicate == 'in':
        attrib_list.append(Portable)
        if entity2:
            entity2.add_attribute(Container)
    elif predicate == 'on':
        attrib_list.append(Portable)
        if entity2:
            entity2.add_attribute(Support)
    elif predicate == 'raw':
        attrib_list.append(Cookable)
    # elif predicate == 'inedible':
    #     entity.del_attribute(Edible)
    elif predicate == 'edible':
        attrib_list.append(Edible)
    elif predicate == 'drinkable':
        attrib_list.append(Drinkable)
    elif predicate == 'cookable':
        attrib_list.append(Cookable)
    elif predicate == 'cooked' \
      or predicate == 'fried' \
      or predicate == 'baked' \
      or predicate == 'toasted' \
      or predicate == 'grilled' \
      or predicate == 'roasted':
        attrib_list.append(Edible)
        attrib_list.append(Cookable)
    elif predicate == 'needs_cooking':
        attrib_list.append(Cookable)
        # assert groundtruth, "Non-GT KG should not receive these facts"
    elif predicate == 'uncut':
        attrib_list.append(Cutable)
    elif predicate == 'cuttable':
        attrib_list.append(Cutable)
    elif predicate in ['chopped', 'sliced', 'diced', 'minced']:
        attrib_list.append(Cutable)
   # if GT mode, accept some ground-truth predicates (to avoid excess warnings)
    elif predicate == 'portable':
        attrib_list.append(Portable)
        assert groundtruth, "Non-GT KG should not receive these facts"
    elif predicate in ['closeable', 'openable']:
        attrib_list.append(Openable)
        assert groundtruth, "Non-GT KG should not receive these facts"
    elif predicate in ['lockable', 'unlockable']:
        attrib_list.append(Lockable)
        assert groundtruth, "Non-GT KG should not receive these facts"
    elif predicate == 'heat_source':
        attrib_list.append(Cooker)
        assert groundtruth, "Non-GT KG should not receive these facts"
    elif predicate in ['fixed', 'holder', 'inedible']:
        assert groundtruth, "Non-GT KG should not receive these facts"
        pass
    else:
        print("Warning -- get_attributes_for_predicate() unexpected predicate:", predicate)
    return attrib_list


def add_attributes_for_type(entity, twvartype):
    attrib_list = get_attributes_for_type(twvartype)
    for attrib in attrib_list:
        entity.add_attribute(attrib)
# GVS 11.dec.2020 added:
        if attrib == Container:
            entity.convert_to_container()
            # entity.state.add_state_variable('openable', 'is_open', '')
            # entity.state.add_state_variable('openable', 'is_open', 'unknown')  # TODO maybe not always (e.g. "in bowl", "in vase"...)
            # entity.state.close()  # ?default to closed
        elif attrib == Support:
            entity.convert_to_support()

    if twvartype == 'd':  # DOOR   Hard-coded specific knowlege about doors (they can potentially be opened)
        if not entity.state.openable:
            entity.state.add_state_variable('openable', 'is_open', '')
            # Indeterminate value until we discover otherwise (via add_attributes_for_predicate)
    elif twvartype == 'oven':  # a container
        # print("ADD ATTRIBS FOR OVEN", entity.state)
        if not entity.state.openable:
            # print("----WAS NOT OPENABLE")
            entity.state.add_state_variable('openable', 'is_open', '')
        elif not entity.state._has_prop('is_open'):
            # print("-----NEED TO ADD is_open")
            # we want to default to closed only once, when creating the entity
            entity.state.add_state_variable('openable', 'is_open', '')
    #     entity.convert_to_container()
    #     # entity.state.add_state_variable('openable', 'is_open', 'unknown')  # TODO maybe not always (e.g. "in bowl", "in vase"...)

def add_attributes_for_predicate(entity, predicate, entity2=None, groundtruth=False):
    attrib_list = get_attributes_for_predicate(predicate, entity2, groundtruth=groundtruth)
    for attrib in attrib_list:
        entity.add_attribute(attrib)

    # inverse relationships
    if predicate == 'in':
        assert entity2 is not None
        entity2.add_attribute(Container)
    elif predicate == 'on':
        assert entity2 is not None
        entity2.add_attribute(Support)

    if predicate == 'inedible':
        entity.del_attribute(Edible)

    # set entity state
    if predicate == 'closed':
        entity.state.close()
    elif predicate == 'open':
        entity.state.open()
    elif predicate == 'locked':
        entity.state.locked()
    elif predicate == 'cooked' \
      or predicate == 'fried' \
      or predicate == 'baked' \
      or predicate == 'toasted' \
      or predicate == 'grilled' \
      or predicate == 'roasted':
        entity.state.cook(cooked_state=predicate)
    elif predicate == 'needs_cooking':
        entity.state.not_cooked()
    elif predicate == 'uncut':
        entity.state.not_cut()
    elif predicate == 'cuttable':
        entity.add_attribute(Cutable)
    elif predicate == 'chopped' \
     or predicate == 'sliced' \
     or predicate == 'diced' \
     or predicate == 'minced':
        entity.state.cut(cut_state=predicate)


def entity_type_for_twvar(vartype):
    if vartype in Entity.entity_types:
        return vartype  # 1:1 mapping, for now
    return None

def twvartype_for_type(entitytype):
    return entitytype  # 1:1 mapping, for now


END_OF_LIST = ';'
START_OF_LIST = ':'
ITEM_SEPARATOR = ','
LINE_SEPARATOR = '\n'

ALWAYS_LIST_EXITS = True
ALWAYS_LIST_ONGROUND = True
ALWAYS_LIST_INVENTORY = True
INCLUDE_ENTITY_TYPES = True


# NOTE: Here's a stove. [...] But the thing is empty. [misleading] (no clue here that stove is a support, not a container)
# on fridge <:> nothing <;>  in oven <:> nothing <;> on table <:> cookbook <;>  on counter <:> red potato <;> on stove: nothing.
# You carry: nothing.

#   # on floor: nothing. )
def describe_visible_objects(loc, inventory_items, exits_descr=None, mention_tracker=None, groundtruth=False, options=None):
    """ Generate a concise (and easily parseable) text of the objects currently observable in the specified room.
         if optional obs_descr string is given, tries to match the order in which entities are mentioned.
     """
    entity_list = []  # will append to and then sort a copy of the list

    # TODO: similarly for exits
    # print(f"entity_list = {entity_list}")
    for entity in loc.entities:  # build index: iterate through original list (plus subentities)
        entity_list.append(entity)
        if mention_tracker:
            mention_tracker.index_mentions(entity.name)
        # print(f"subentites of {entity} = {entity.entities}")
        for subentity in entity.entities:
            if subentity not in entity_list:
                entity_list.append(subentity)  # keep track of all encountered entities & subentities
            if mention_tracker:
                mention_tracker.index_mentions(subentity.name)
    # print(f"Inventory: {self.inventory.entities}")
    for entity in inventory_items:
        if mention_tracker:
            mention_tracker.index_mentions(entity.name)
    # TODO: ?add non-door exits...
    non_door_entities = []
    door_entities = []
    for entity in entity_list:
        if not entity.is_a(DOOR) and not entity.is_a(PERSON) and \
            not entity.parent.is_a(INVENTORY):  # filter out doors, the Player, items in inventory
                non_door_entities.append(entity)
        else:
            door_entities.append(entity)
    on_floor_entities = []
    # filter out portable entities that are on the floor, into a separate list...
    for entity in non_door_entities:
        if Portable in entity.attributes and entity.parent == loc:
            on_floor_entities.append(entity)
    for entity in on_floor_entities:
        non_door_entities.remove(entity)

    if mention_tracker:
        non_door_entities = mention_tracker.sort_entity_list_by_first_mention(non_door_entities)
        # inventory_items = mention_tracker.sort_entity_list_by_first_mention(inventory_items)
        on_floor_entities = mention_tracker.sort_entity_list_by_first_mention(on_floor_entities)
    # else:  # should already be sorted by containment hierarchy...
    #     print("non_door_entities:", non_door_entities)
    #     print("on_floor_entities:", on_floor_entities)
    #     non_door_entities = sort_entity_list_by_parent(loc, non_door_entities)
    #     on_floor_entities = sort_entity_list_by_parent(loc, on_floor_entities)

    idx = 0
    obs_descr = ""   #"You see:"+LINE_SEPARATOR
    while idx < len(non_door_entities):
        descr_str, idx = _format_entities_descr(non_door_entities, idx, groundtruth=groundtruth, options=options)
        obs_descr += descr_str + LINE_SEPARATOR
    if exits_descr:
        obs_descr += LINE_SEPARATOR + exits_descr
    if on_floor_entities:
        onground_descr = f" {ITEM_SEPARATOR} ".join([_format_entity_name(item) for item in on_floor_entities])
    else:
        onground_descr = "nothing" if ALWAYS_LIST_ONGROUND else ''
    if onground_descr:
        obs_descr += LINE_SEPARATOR + f"ON floor {START_OF_LIST} {onground_descr} {END_OF_LIST}"
    if len(inventory_items):
        inventory_descr = f"Carrying {START_OF_LIST} " + f" {ITEM_SEPARATOR} ".join([_format_entity_name(item) for item in inventory_items])
    else:
        inventory_descr = f"Carrying {START_OF_LIST} nothing" if ALWAYS_LIST_INVENTORY else ''
    if inventory_descr:
        obs_descr += LINE_SEPARATOR + inventory_descr + ' ' + END_OF_LIST
    return obs_descr


def _format_entity_name(entity):
    out_str = ""
    if entity.state:
        state_descr = entity.state.format_descr()
        if state_descr:
            out_str += state_descr
    if INCLUDE_ENTITY_TYPES and entity._type:
        out_str += f"_{entity._type}_ "
    out_str += f"{entity.name}"
    return out_str


def _format_entities_descr(entity_list, idx, groundtruth=False, options=None):
    entity = entity_list[idx]
    idx += 1
    if entity.is_container:
        descr_str = f"IN {_format_entity_name(entity)} {START_OF_LIST} "
        held_entities = entity.entities
    elif entity.is_support:
        descr_str = f"ON {_format_entity_name(entity)} {START_OF_LIST} "
        held_entities = entity.entities
    else:
        descr_str = f"{_format_entity_name(entity)}"
        return descr_str, idx

    if not groundtruth and entity.is_container \
        and entity.state.openable \
        and entity.state._has_prop('is_open') \
        and not entity.state.is_open \
        and (not entity.has_been_opened or options == 'parsed-obs'):
            # if we've never looked inside or we're not using Oracle knowledge about what we've already seen)
            descr_str += f"unknown {END_OF_LIST}"  # then we don't know what's in there
            return descr_str, idx

    if not held_entities or len(held_entities) == 0:
        descr_str += f"nothing {END_OF_LIST}"
    else:
        first = True
        while idx < len(entity_list) and entity_list[idx] in held_entities:
            if not first:
                descr_str += f" {ITEM_SEPARATOR} "
            first = False
            descr_str += _format_entity_name(entity_list[idx])
            idx += 1
        descr_str += f" {END_OF_LIST}"
    return descr_str, idx


class KnowledgeGraph:
    """
    Knowledge Representation consists of visisted locations.

    """
    def __init__(self, event_stream, groundtruth=False, logger=None, debug=False):
        self._logger = logger
        self._debug = debug
        self._formatting_options = 'kg-descr'   # or = 'parsed-obs':
        self._unknown_location   = UnknownLocation()
        self._nowhere            = Location(name="NOWHERE", entitytype=NONEXISTENCE_LOC)
        self._player             = Person(name="You", description="The Protagonist")
        self._locations          = set()   # a set of top-level locations (rooms)
        self._connections        = ConnectionGraph()  # navigation links connecting self._locations
        self.event_stream        = event_stream
        self.groundtruth         = groundtruth   # bool: True if this is the Ground Truth knowledge graph
        self._entities_by_name   = {}   # index: name => entity (many to one: each entity might have more than one name)

        self._unknown_location.add_entity(self._player)

        self._update_entity_index(self._unknown_location)
        self._update_entity_index(self._player)
        self._update_entity_index(self._player.inventory)  # add the player's inventory to index of Locations

    def set_logger(self, logger):
        self._logger = logger

    def dbg(self, msg):
        if self._logger:
            logger = self._logger
            logger.debug(msg)
        else:
            print("### DEBUG:", msg)

    def warn(self, msg):
        if self._logger:
            logger = self._logger
            logger.warning(msg)
        else:
            print("### WARNING:", msg)

    def broadcast_event(self, ev):
        if self.event_stream:
            self.event_stream.push(ev)
        else:
            if self._debug:
                print("KG: No Event Stream! event not sent:", ev)

    # ASSUMPTION: names are stable. A given name will always refer to the same entity. Entity names don't change over time.
    def _update_entity_index(self, entity):
        for name in entity.names:
            if name in self._entities_by_name:
                assert self._entities_by_name[name] is entity, \
                    f"name '{name}' should always refer to the same entity"
            self._entities_by_name[name] = entity

    def __str__(self):
        s = "Knowledge Graph{}\n".format('[GT]' if self.groundtruth else '')
        if self.player_location == None:
            s += "PlayerLocation: None"
        else:
            s += "PlayerLocation: {}".format(self.player_location.name)
        s += "\n" + str(self.inventory)
        s += "\nKnownLocations:"
        for loc in self.locations:
                s += "\n" + loc.to_string("  ")
                outgoing = self.connections.outgoing(loc)
                if outgoing:
                    s += "\n    Connections:"
                    for con in outgoing:
                        s += "\n      {} --> {}".format(con.action, con.to_location.name)
        return s


    def set_formatting_options(self, formatting_options:str):
        """ options that control some nuances of the output from describe_room(), describe_exits(), etc """
        assert formatting_options == 'parsed-obs' or formatting_options == 'kg-descr', formatting_options
        prev_options = self._formatting_options
        self._formatting_options = formatting_options
        return prev_options

    def describe_exits(self, loc, mention_tracker=None):
        outgoing_directions = self.connections.outgoing_directions(loc)
        if not ALWAYS_LIST_EXITS and not outgoing_directions:
            return ""
        if mention_tracker:
            outgoing_directions = mention_tracker.sort_entity_list_by_first_mention(outgoing_directions)
# NOTE 2020-12-24: changed ---Exits--- => Exits, because tokenizer splits ---Exits--- it into 3 tokens
        exits_list = [
            self.connections.connection_for_direction(loc, direction).to_string(options=self._formatting_options)
            for direction in outgoing_directions
        ]
        if not exits_list:    # if there are no exits,
            exits_list = ["none"]
        return "Exits : " + f" {ITEM_SEPARATOR} ".join(exits_list) + f" {END_OF_LIST}{LINE_SEPARATOR}"

    def describe_room(self, roomname, obs_descr=None):
        """ Generate a concise (and easily parseable) text description of the room and what's currently observable there.
            if optional obs_descr string is given, tries to match the order in which entities are mentioned.
        """
        mention_tracker = None if not obs_descr else MentionIndex(obs_descr)
        if mention_tracker:
            mention_tracker.index_mentions('north')  # describe_exits will need these to sort exits
            mention_tracker.index_mentions('east')
            mention_tracker.index_mentions('south')
            mention_tracker.index_mentions('west')

        loc = self.get_location(roomname)
        if not loc or (not loc.is_a(ROOM) and loc.is_a(ROOM) is not None):  # is_a can return None if entitytype is unset
            return "-= Unknown Location =-"
        elif loc == self._unknown_location:
            return "-= Unknown Location =-"
        #else:   # we've got a genuine Room object
        room_descr = f"-= {loc.name} =-\n"

        # if loc == self.player_location:
        #     room_descr += f"You are_in: {loc.name}\n"
        if loc == self.player_location:
            inventory_items = list(self.inventory.entities)  # pass a copy, for safey
        else:
            inventory_items = []
        exits_descr = self.describe_exits(loc, mention_tracker=mention_tracker)
        room_descr += describe_visible_objects(loc, inventory_items, exits_descr=exits_descr,
                                               mention_tracker=mention_tracker, groundtruth=self.groundtruth,
                                               options=self._formatting_options)
        return room_descr

    @property
    def locations(self):
        return self._locations

    def add_location(self, new_location: Location) -> NewLocationEvent:
        """ Adds a new location object and broadcasts a NewLocation event. """
        is_new = new_location not in self._locations
        self._update_entity_index(new_location)
        if is_new and new_location.parent is None and new_location is not self._unknown_location:
            self._locations.add(new_location)
            return NewLocationEvent(new_location, groundtruth=self.groundtruth)
        return None

    def locations_with_name(self, location_name):
        """ Returns all locations with a particular name. """
        return [loc for loc in self.locations if loc.has_name(location_name)]

    @property
    def inventory(self):
        return self._player.inventory

    @property
    def player_location(self):
        return self._player.location

    # @player_location.setter
    def set_player_location(self, new_location):
        """ Changes player location and broadcasts a LocationChangedEvent. """
        prev_location = self._player.location
        # print(f"set_player_location: {prev_location} {self._player.location} {new_location}", prev_location.entities)
        if new_location == prev_location:
            return False
        if new_location:
            newly_discovered = new_location.visit()
            if newly_discovered and self._debug:
                print(f"visit() DISCOVERED {self}")
            # if not Location.is_unknown(prev_location):  # Don't bother broadcasting spawn location
            #     self.broadcast_event(LocationChangedEvent(new_location))
            #IMPORTANT: this is the correct way to set location for an entity
            new_location.add_entity(self._player)  #THIS IS WRONG: player.location = new_location
        return True

    @property
    def connections(self):
        return self._connections

    def add_connection(self, new_connection, with_inverse=False):
        """ Adds a connection object.
         Does nothing if a similar connection already exists"""
        self._connections.add(new_connection, self, assume_inverse=with_inverse)
        #TODO DEBUG: self._connections.add(new_connection, self, assume_inverse=False)

    def reset(self):
        """Returns the knowledge_graph to a state resembling the start of the
        game. Note this does not remove discovered objects or locations. """
        kg = self
        if self._debug:
            print(f"RESETTING Knowledge Graph {kg}")
        # self.inventory.reset(kg)  -- done in _player.reset()
        # print("***** reset _player")
        self._player.reset(kg)
        # print("***** reset locations")
        for location in self.locations:
            location.reset(kg)

    def is_object_portable(self, objname: str) -> bool:
        entity = self.get_entity(objname)
        return entity is not None and Portable in entity.attributes

    def is_object_cut(self, objname: str, verb: str) -> bool:
        entity = self.get_entity(objname)
        return entity is not None and entity.state.cuttable \
               and entity.state.is_cut and entity.state.is_cut.startswith(verb)

    def is_object_cooked(self, objname: str, verb: str) -> bool:
        is_cooked = False
        entity = self.get_entity(objname)
        if entity is not None and entity.state.cookable:
            cooked_state = entity.state.is_cooked
            if cooked_state:
                 is_cooked = (cooked_state.startswith(verb) or
                    cooked_state == 'fried' and verb == 'fry')
        return is_cooked

    def entities_with_name(self, entityname, entitytype=None):
        """ Returns all entities with a particular name. """
        ret = set()
        if entityname in self._entities_by_name:
            ret.add(self._entities_by_name[entityname])
        # for loc in chain(self._locations, [self.inventory]):
        #     e = loc.get_entity_by_name(entityname)
        #     if e:
        #         ret.add(e)
        # if len(ret) == 0:  # check to see if entity has been previously mentioned, but not yet found
        #     e = self._unknown_location.get_entity_by_name(entityname)
        #     if e:
        #         ret.add(e)
        if entitytype:
            wrong_type = {e for e in ret if e._type != entitytype}
            if wrong_type:
                self.warn(f"Filtering out entities with non-matching type != {entitytype}: {wrong_type}")
                return ret - wrong_type
        return ret

    def where_is_entity(self, entityname, entitytype=None, allow_unknown=True, top_level=False):
        """ Returns a set of (shallow) locations where an entity with a specific name can be found """
        entityset = self.entities_with_name(entityname, entitytype=entitytype)
        if top_level:
            ret = {self.primogenitor(entity.location) for entity in entityset}
        else:
            ret = {entity.location for entity in entityset}

        if not allow_unknown:
            ret.discard(self._unknown_location)
        return ret

    def location_of_entity_with_name(self, entityname, entitytype=None, allow_unknown=False):
        """ Returns a single (top-level: =ROOM) location where an entity with a specific name can be found """
        location_set = self.where_is_entity(entityname, entitytype=entitytype, allow_unknown=allow_unknown)
        room_set = {self.primogenitor(loc) for loc in location_set}
        # print(f"DEBUG location_of_entity({entityname}:{entitytype}) => {location_set}")
        if not allow_unknown:
            room_set.discard(self._unknown_location)
        if room_set:
            if len(room_set) > 1:
                self.warn(f"multiple locations for <{entityname}>: {location_set} => {room_set}")
            return room_set.pop()  # choose one (TODO: choose with shortest path to player_location)
        return None

    def location_of_entity_is_known(self, entityname: str) -> bool:
        loc_set = self.where_is_entity(entityname, allow_unknown=False, top_level=True)
        return len(loc_set) > 0

    def path_to_unknown(self):
        return self.connections.shortest_path(self.player_location, self._unknown_location)

    def get_holding_entity(self, entity):
        """ Returns container (or support) where an entity with a specific name can be found """
        if entity.location != self._unknown_location:
            parent = entity.location
            if parent:
                if parent._type == CONTAINED_LOCATION or parent._type == SUPPORTED_LOCATION:
                    parent = parent.parent
                    assert parent._type == CONTAINER or parent._type == SUPPORT
                    return parent
        return None

    def primogenitor(self, entity):
        """ Follows a chain of parent links to an ancestor with no parent """
        ancestor = entity
        while ancestor and ancestor.parent and ancestor.parent is not ancestor:
            ancestor = ancestor.parent
        return ancestor

    def add_entity_at_location(self, entity, location):
        if location.add_entity(entity):
            if self.event_stream and not self.groundtruth:
                ev = NewEntityEvent(entity)
                self.broadcast_event(ev)

    def act_on_entity(self, action, entity, rec: ActionRec):
        if entity.add_action_record(action, rec) and rec.p_valid > 0.5 and not self.groundtruth:
            ev = NewActionRecordEvent(entity, action, rec.result_text)
            self.broadcast_event(ev)

    def action_at_current_location(self, action, p_valid, result_text):
        loc = self.player_location
        loc.action_records[action] = ActionRec(p_valid, result_text)
        if not self.groundtruth:
            ev = NewActionRecordEvent(loc, action, result_text)
            self.broadcast_event(ev)

    def get_location(self, roomname, create_if_notfound=False, entitytype=ROOM):
        locations = self.locations_with_name(roomname)
        if locations:
            assert len(locations) == 1
            return locations[0]
        elif create_if_notfound:
            new_loc = Location(name=roomname, entitytype=entitytype)
            ev = self.add_location(new_loc)
            # if self.groundtruth: DISCARD NewLocationEvent else gi.broadcast_event(ev)
            if self.groundtruth:
                new_loc._discovered = True   # HACK for GT: all locations (except the UnknownLocation) are known
            else:  # not self.groundtruth:
                assert ev, "Adding a newly created Location should return a NewLocationEvent"
                self.broadcast_event(ev)
            # if not self.groundtruth:
            #     print("created new {}Location:".format('GT ' if self.groundtruth else ''), new_loc)
            return new_loc
        if self._debug:
            print("LOCATION NOT FOUND:", roomname)
        return None

    def maybe_move_entity(self, entity, locations=None):
        assert locations
        # might need to move it from wherever it was to its new location
        if not isinstance(entity, Door):
            assert len(locations) == 1, f"{locations}"
        else:
            assert len(locations) <= 2, f"{locations}"

        loc_set = set(locations)
        for loc in loc_set:
            if loc.add_entity(entity):
                if not self.groundtruth and not Location.is_unknown(loc):
                    if entity.location == entity._init_loc:
                        if self._debug:
                            print(f"\tDISCOVERED NEW entity: {entity} at {loc}")
                        ev = NewEntityEvent(entity)
                        self.broadcast_event(ev)
                    else:
                        if self._debug:
                            print(f"\tMOVED entity: {entity} to {loc}")

    def get_entity(self, name, entitytype=None):
        if isinstance(name, Entity):    # shortcut search by name if we already have an entity
            entity = name
            name = entity.name
            if entitytype is None or entity._type == entitytype:
                return entity
        entities = self.entities_with_name(name, entitytype=entitytype)
        if entities:
            if len(entities) > 1:
                self.warn(f"get_entity({name}, entitytype={entitytype}) found multiple potential matches: {entities}")
            for e in entities:
                if entitytype is None or e._type == entitytype:
                    return e   # if more than one: choose the first matching entity
        return None

    def create_new_object(self, name, entitytype, description=''):
        initial_loc = self._unknown_location
        if entitytype == DOOR:
            new_entity = Door(name=name, description=description)
        else:
            new_entity = Thing(name=name, description=description, entitytype=entitytype)  # location=initial_loc,
        added_new = initial_loc.add_entity(new_entity)
        assert added_new   # since this is a new entity, there shouldn't already be an entity with the same name
        add_attributes_for_type(new_entity, entitytype)
        self._update_entity_index(new_entity)
        return new_entity

    def add_obj_to_obj(self, fact, maybe_new_entities_list, rel=None):
        assert rel == fact.name
        # rel = fact.name
        o = fact.arguments[0]
        h = fact.arguments[1]
        if o.name.startswith('~') or h.name.startswith('~'):
            if self._debug:
                print("_add_obj_to_obj: SKIPPING FACT", fact)
            return None, None
        assert h.type != 'I'   # Inventory

        entitytype = entity_type_for_twvar(o.type)
        obj = self.get_entity(o.name, entitytype=entitytype)
        if not obj:
            obj = self.create_new_object(o.name, entitytype)  #, locations=loc_list)
            maybe_new_entities_list.append(obj)
            if not self.groundtruth:
                if self._debug:
                    print("\tADDED NEW Object {} :{}: {}".format(obj, fact.name, h.name))
                # print("DISCOVERED NEW entity:", obj)
                ev = NewEntityEvent(obj)
                self.broadcast_event(ev)

        holder = self.get_entity(h.name, entitytype=entity_type_for_twvar(h.type))
        if holder:
            if rel == 'in':
                self._update_entity_index(
                    holder.convert_to_container())
            elif rel == 'on':
                self._update_entity_index(
                    holder.convert_to_support())
            holder.add_entity(obj, self, rel=rel)
        return obj, holder

    def add_obj_to_inventory(self, fact, maybe_new_entities_list):
        # rel = fact.name
        o = fact.arguments[0]
        h = fact.arguments[1]
        if o.name.startswith('~'):
            assert False, f"_add_obj_to_inventory: UNEXPECTED FACT {fact}"
        assert h.name == 'I'   # Inventory
        loc_list = [self.inventory]  # a list containing exactly one location

        entitytype = entity_type_for_twvar(o.type)
        obj = self.get_entity(o.name, entitytype=entitytype)
        if not obj:
            obj = self.create_new_object(o.name, entitytype)  #, locations=loc_list)
            maybe_new_entities_list.append(obj)
            if self._debug:
                print("\tADDED NEW Object {} :{}: {}".format(obj, fact.name, h.name))
        self.maybe_move_entity(obj, locations=loc_list)
        # DEBUGGING: redundant SANITY CHECK
        if not self.inventory.get_entity_by_name(obj.name):
            self.warn(f"{obj.name} NOT IN INVENTORY {self.inventory.entities}")
            assert False
        return obj

    def update_facts(self, obs_facts, prev_action=None):
        if self._debug:
            print(f"*********** {'GROUND TRUTH' if self.groundtruth else 'observed'} FACTS *********** ")
        player_loc = None
        door_facts = []
        at_facts = []
        on_facts = []
        in_facts = []
        inventory_facts = []
        inventory_items = []
        maybe_new_entities = []
        other_facts = []
        for fact in obs_facts:
            a0 = fact.arguments[0]
            a1 = fact.arguments[1] if len(fact.arguments) > 1 else None
            if fact.name == 'link' and self.groundtruth:
                door_facts.append(fact)
            elif fact.name == 'at':
                if a0.type == 'P' and a1.type == 'r':
                    player_loc = self.get_location(a1.name, create_if_notfound=True)
                    if player_loc not in maybe_new_entities:
                        maybe_new_entities.append(player_loc)
                else:
                    at_facts.append(fact)
            elif fact.name == 'on':
                on_facts.append(fact)
            elif fact.name == 'in':
                if a1 and a1.type == 'I':  # Inventory
                    inventory_facts.append(fact)
                else:
                    in_facts.append(fact)
            elif fact.name in DIRECTION_ACTIONS:
                if a0.type == 'r' and a1.type == 'r' and self.groundtruth:
                    # During this initial pass we create locations and connections among them
                    # print('++CONNECTION:', fact)
                    loc0 = self.get_location(a0.name, create_if_notfound=True)
                    if loc0 not in maybe_new_entities:
                        maybe_new_entities.append(loc0)
                    loc1 = self.get_location(a1.name, create_if_notfound=True)
                    if loc1 not in maybe_new_entities:
                        maybe_new_entities.append(loc1)
                    door = None  # will add info about doors later
                    new_connection = Connection(loc1, DIRECTION_ACTIONS[fact.name], loc0, doorway=door)
                    self.add_connection(new_connection, with_inverse=True)  # does nothing if connection already present
                elif (a0.type == 'd' or a0.type == 'e') and a1.type == 'r':
                    # print("\t\tdoor fact -- ", fact)
                    door_facts.append(fact)
                    # pass
                else:
                    # print("--IGNORING DIRECTION_FACT:", fact)
                    pass
            else:
                other_facts.append(fact)
        # 2nd pass, add doors to connections
        for fact in door_facts:
            r0 = None
            r1 = None
            d = None
            door_locations = []
            if len(fact.arguments) == 3:
                r0 = fact.arguments[0]
                d = fact.arguments[1]
                r1 = fact.arguments[2]
                assert fact.name == 'link'
                assert r0.type == 'r'
                assert r1.type == 'r'
                assert d.type == 'd'
                if not self.groundtruth:
                    assert False, f"UNEXPECTED: link fact in non-groundtruth door_facts: {fact}"
                    continue   # link
            elif len(fact.arguments) == 2:
                r0 = fact.arguments[1]
                d = fact.arguments[0]
                assert fact.name in DIRECTION_ACTIONS
                assert r0.type == 'r'
                assert d.type == 'd' or d.type == 'e'
                if self.groundtruth:
                    continue   # rely on 'link' facts to create doors
            if r0:
                loc0 = self.get_location(r0.name, create_if_notfound=False)
                door_locations.append(loc0)
            if r1:
                loc1 = self.get_location(r1.name, create_if_notfound=False)
                door_locations.append(loc1)
            else:
                door_locations.append(self._unknown_location)  # we don't yet know what's on the other side of the door

            if d.type == 'e':  # an exit without a door
                door = None
            else:
                door = self.get_entity(d.name, entitytype=DOOR)
                if not door:
                    door = self.create_new_object(d.name, DOOR)  #, locations=door_locations)
                    maybe_new_entities.append(door)
                self.maybe_move_entity(door, locations=door_locations)
            if fact.name in DIRECTION_ACTIONS:
                direction = fact.name
                if door:
                    door.add_direction_rel(ConnectionRelation(from_location=loc0, direction=direction))
                if not self.groundtruth:
                    door_connection = Connection(loc0, DIRECTION_ACTIONS[fact.name], self._unknown_location, doorway=door)
                    self.add_connection(door_connection, with_inverse=True)  # does nothing if connection already present
            if r1:  #len(door_locations) == 2:
                assert self.groundtruth
                # NOTE: here we rely on the fact that we added a connection while processing GT DIRECTION_ACTIONS above
                linkpath = self.connections.shortest_path(loc0, loc1)
                assert len(linkpath) == 1
                connection = linkpath[0]
                connection.doorway = door  # add this DOOR to the Connection
            if self.groundtruth and door:
                door.state.open()   # assume that it's open, until we find a closed() fact...
        # for fact in other_facts:
        #     if fact.name == 'closed' and fact.arguments[0].type == 'd':
        #         doorname = fact.arguments[0].name
        #         doors = self.entities_with_name(doorname, entitytype=DOOR)
        #         assert len(doors) == 1
        #         door = list(doors)[0]
        #         door.state.close()

        if player_loc:  # UPDATE player_location
            prev_loc = self.player_location
            if self.set_player_location(player_loc):
                if self.groundtruth:
                    if self._debug:
                        print(f"GT knowledge graph updating player location from {prev_loc} to {player_loc}")
                else:
                    if prev_action:
                        if self._debug:
                            print(f"Action: <{prev_action}> CHANGED player location from {prev_loc} to {player_loc}")
                        if prev_action and player_loc != self._unknown_location and prev_loc != self._unknown_location:
                            if isinstance(prev_action, Action):
                                prev_action_words = prev_action.text().split()
                            else:
                                prev_action_words = prev_action.split()
                            verb = prev_action_words[0]
                            if verb == 'go':
                                verb = prev_action_words[1]
                            direction_rel = verb + '_of'
                            if direction_rel not in DIRECTION_ACTIONS:
                                self.warn(f"WARNING: UNEXPECTED direction: {verb} (maybe '{prev_action}' is not a NavAction?)")
                            else:
                                door = None   # TODO: remember door directions relative to rooms, and look up the appropr door
                                new_connection = Connection(prev_loc,
                                                            DIRECTION_ACTIONS[direction_rel],
                                                            player_loc,
                                                            doorway=door)
                                self.add_connection(new_connection, with_inverse=True)
                # if not Location.is_unknown(player_loc):
        for fact in at_facts:
            if self._debug:
                print("DEBUG at_fact:", fact)
            o = fact.arguments[0]
            r = fact.arguments[1]
            loc = self.get_location(r.name, create_if_notfound=False)
            # print("DEBUG at_facts: LOCATION for roomname", r.name, "->", loc)
            if loc:
                locs = [loc]
            else:
                self.warn(f"at_facts failed to find location {r.name}")  # ({self.player_location}, {self.locations})
                locs = None
            if r.type == 'r':
                entitytype = entity_type_for_twvar(o.type)
                obj = self.get_entity(o.name, entitytype=entitytype)
                if not obj:
                    obj = self.create_new_object(o.name, entitytype)  #, locations=locs)
                    #add_attributes_for_type(obj, o.type)   # done by create_new_object
                    maybe_new_entities.append(obj)
                self.maybe_move_entity(obj, locations=locs)
            else:
                self.warn(f"-- ADD FACTS: unexpected location for at(o,l): {r}")
            # add_attributes_for_type(obj, o.type)  # GVS 2020-12-12 This was redundant here, and potentially dangerous

        #NOTE: the following assumes that objects are not stacked on top of other objects which are on or in objects
        # and similarly, that "in" relations are not nested.
        #TODO: this should be generalized to work correctly for arbitrary chains of 'on' and 'in'
        #(currently assumes only one level: container or holder is immobile, previously identified "at" a place)

        for fact in on_facts:
            if self._debug:
                print("DEBUG on_fact:", fact)
            o1, o2 = self.add_obj_to_obj(fact, maybe_new_entities, rel='on')
            if o1 and o2:
                add_attributes_for_predicate(o1, 'on', o2, groundtruth=self.groundtruth)
        for fact in in_facts:
            if self._debug:
                print("DEBUG in_fact:", fact)
            o1, o2 = self.add_obj_to_obj(fact, maybe_new_entities, rel='in')
            if o1 and o2:
                add_attributes_for_predicate(o1, 'in', o2, groundtruth=self.groundtruth)
        for fact in inventory_facts:
            if self._debug:
                print("DEBUG inventory fact:", fact)
            o1 = self.add_obj_to_inventory(fact, maybe_new_entities)
            if o1 not in inventory_items:
                inventory_items.append(o1)

        for fact in other_facts:
            predicate = fact.name
            if predicate == 'cooking_location' \
            or predicate.startswith('ingredient_') \
            or predicate == 'free' \
            or predicate == 'base' \
            or predicate == 'out':
                continue

            a0 = fact.arguments[0]
            o1 = None
            o2 = None
            if a0.name.startswith('~'):
                continue
            if len(fact.arguments) > 1:
                a1 = fact.arguments[1]
            else:
                a1 = None
            if self._debug:
                print("DEBUG other fact:", fact)
            o1 = self.get_entity(a0.name, entitytype=entity_type_for_twvar(a0.type))
            if a1:
                o2 = self.get_entity(a1.name, entitytype=entity_type_for_twvar(a1.type))
            if o1:
                add_attributes_for_predicate(o1, predicate, entity2=o2, groundtruth=self.groundtruth)
            else:
                if predicate == 'edible' and a0.name == 'meal':
                    continue
                self.warn(f"add_attributes_for_predicate {predicate}, didnt find an entity corresponding to {a0}")
        if prev_action:
            self.special_openclose_oven(prev_action)
        for obj in maybe_new_entities:
            if Cookable in obj.attributes:
                if not obj.state.cookable:
                    if self._debug:
                        print(f"update_facts: Setting raw({obj.name})")
                    obj.state.not_cooked()  # a cookable (food) item, but we didn't see a cooked(x) Proposition => raw(x)
        self.handle_gone_from_inventory(inventory_items)
        if self._debug:
            print(f"---------------- update FACTS end -------------------")

    def handle_gone_from_inventory(self, in_inventory_items, prev_action=None):
        # if self._debug:
        #     for item in in_inventory_items:
        #         print("update_facts -- in inventory:", item)
        gone_items = []
        for item in self.inventory:   # at this point we should have seen an 'at' fact for every item in our inventory
            if item not in in_inventory_items:
                # assume it was consumed by the previous action
                gone_items.append(item)

        if self._debug:
            print("update_facts -- disappeared from inventory:", gone_items)
        for item in gone_items:
            self._nowhere.add_entity(item)   # move it from inventory to NOWHERE (non-existence)

    def special_openclose_oven(self, prev_action):
        if isinstance(prev_action, Action):
            prev_action_words = prev_action.text().split()
        else:
            prev_action_words = prev_action.split()
        if prev_action_words[0] in ['open', 'close'] and 'oven' in prev_action_words:
            if self._debug:
                print(f"SPECIAL CASE FOR prev_action: {prev_action_words}")
            oven = self.player_location.get_entity_by_name('oven')
            if oven:
                if prev_action_words[0] == 'open':
                    oven.open()
                elif prev_action_words[0] == 'close':
                    oven.close()


class ConnectionGraph:
    """
    Graph of connections between locations.

    """
    def __init__(self):
        self._out_graph = {}  # {from_Location : {direction: Connection(outgoing)} }
        self._in_graph  = {}  # {to_Location : { tuple(from_Location, direction): Connection(incoming)} }

    def add(self, connection, kg: KnowledgeGraph, assume_inverse=False):
        """ Adds a new connection to the graph if it doesn't already exist,
            or updates an existing connection (if to_location was previously UnknownLocation)"""
        from_location = connection.from_location
        assert not Location.is_unknown(from_location)
        to_location = connection.to_location
        direction = map_action_to_direction(connection.action)
        added_new = []
        # kg.broadcast_event(NewConnectionEvent(connection, groundtruth=kg.groundtruth))
        if from_location not in self._out_graph:
            added_new.append(f"out_graph[{from_location.name}]")
            self._out_graph[from_location] = {direction: connection}
        else:
            if direction in self._out_graph[from_location]:
                # if self._out_graph[from_location][direction] != connection:
                    # print(f"... updating {self._out_graph[from_location][direction]} <= {connection}")
                connection = self._out_graph[from_location][direction].update(connection)
            else:
                added_new.append(f"out_graph[{from_location.name}][{direction}]")
                self._out_graph[from_location][direction] = connection

        if not Location.is_unknown(to_location):   # don't index connections incoming to UnknownLocation
            incoming_rel = ConnectionRelation(from_location=from_location, direction=direction)
            if to_location not in self._in_graph:
                added_new.append(f"in_graph[{to_location.name}] {incoming_rel}")
                self._in_graph[to_location] = {incoming_rel: connection}
            else:
                if incoming_rel in self._in_graph[to_location]:
                    # if self._in_graph[to_location][incoming_rel] != connection:
                    #     print(f"... updating {self._in_graph[to_location][incoming_rel]} <= {connection}")
                    self._in_graph[to_location][incoming_rel].update(connection)
                else:
                    added_new.append(f"in_graph[{to_location.name}][{incoming_rel}]")
                    self._in_graph[to_location][incoming_rel] = connection

        # if there's a doorway associated with this connection,
        # we might be able to use its 2nd endpoint to update or create a reverse connection
        # (going back the other way through the same door)
        if connection.doorway and \
                connection.doorway.direction_from_loc1 and \
                connection.doorway.direction_from_loc2 and \
                not Location.is_unknown(connection.doorway.direction_from_loc2.from_location):
            assert not Location.is_unknown(connection.doorway.direction_from_loc1.from_location)
            rev_direction = None
            if connection.doorway.direction_from_loc1.from_location == connection.from_location:
                if connection.doorway.direction_from_loc1.direction:
                    assert connection.doorway.direction_from_loc1.direction == direction
                to_location = connection.from_location
                from_location = connection.doorway.direction_from_loc2.from_location
                rev_direction = connection.doorway.direction_from_loc2.direction
            else:
                assert connection.doorway.direction_from_loc2.from_location == connection.from_location
                if connection.doorway.direction_from_loc2.direction:
                    assert connection.doorway.direction_from_loc2.direction == direction
                if Location.is_unknown(connection.to_location):  # maybe fill it in using doorway info
                    to_location = connection.from_location
                    from_location = connection.doorway.direction_from_loc1.from_location
                    rev_direction = connection.doorway.direction_from_loc1.direction
            if rev_direction:
                reverse_connection = Connection(from_location,
                                                DIRECTION_ACTIONS[rev_direction],
                                                to_location=to_location,
                                                doorway=connection.doorway)
                self.add(reverse_connection, kg, assume_inverse=False)
                assume_inverse = False   # since we've just built a reverse connection, skip default inverse logic

        # if added_new and not kg.groundtruth:
        #         print("\tADDED NEW {}CONNECTION".format('GT ' if kg.groundtruth else ''), added_new, connection)

        if assume_inverse:  # assume that 180 inverse direction connects to_location => from_location
            if not Location.is_unknown(connection.to_location):
                reverse_connection = Connection(connection.to_location,
                                                DIRECTION_ACTIONS[reverse_direction(direction)],
                                                to_location=from_location,
                                                doorway=connection.doorway)
                self.add(reverse_connection, kg, assume_inverse=False)

    def incoming(self, location):
        """ Returns a list of incoming connections to the given location. """
        if location in self._in_graph:
            return list(self._in_graph[location].values())
        else:
            return []

    def outgoing(self, location):
        """ Returns a list of outgoing connections from the given location. """
        if location in self._out_graph:
            return list(self._out_graph[location].values())
        else:
            return []

    def outgoing_directions(self, location):
        if location not in self._out_graph:
            return []
        return list(map(simple_direction, self._out_graph[location].keys()))

    def connection_for_direction(self, location, direction):   # returns None if no exit that direction
        if not direction.endswith('_of'):
            direction = direction+'_of'    # convert e.g. north => north_of
        return self._out_graph[location].get(direction, None)

    def navigate(self, location, nav_action):
        """Returns the destination that is reached by performing nav_action
        from location.
        """
        if not isinstance(nav_action, Action):
            raise ValueError("Expected Action. Got {}".format(type(nav_action)))
        for connection in self.outgoing(location):
            if connection.action == nav_action:
                return connection.to_location
        return None

    def shortest_path(self, start_location, end_location, bestpath=None, visited=None):
        """ Find the shortest path between start and end locations. """
        if bestpath is None:
            bestpath = []
        if visited is None:
            visited = []
        if start_location == end_location:
            return bestpath
        if start_location not in self._out_graph:
            return None
        if start_location in visited:
            # print(f"shortest_path: SKIPPING {start_location} to avoid looping")
            return None
        visited.append(start_location)
        shortest = None
        for connection in self.outgoing(start_location):
            if connection not in bestpath:
                newpath = self.shortest_path(connection.to_location,
                                             end_location,
                                             bestpath + [connection], visited=visited)
                if newpath:
                    if not shortest or len(newpath) < len(shortest):
                        shortest = newpath
        return shortest

    def to_string(self):
        strout = "Connection Graph:\n\n"
        strout += "    outgoing:\n"
        for loc in self._out_graph:
            for con in self.outgoing(loc):
                strout += con.to_string()+'\n'
        strout += "\n    incoming:\n"
        for loc in self._in_graph:
            for con in self.incoming(loc):
                strout += con.to_string()+'\n'
        return strout

    def __str__(self):
        return self.to_string()


def _format_doorinfo(doorentity, options=None):
    if doorentity is None:
        return ''
    if options == 'kg-descr' or options == 'parsed-obs':
        if doorentity.state.openable and not doorentity.state.is_open:
            return f" +closed {doorentity.name}"
        return f" +open {doorentity.name}"
    #else:
    if doorentity.state.openable and not doorentity.state.is_open:
        return f"{doorentity.name}(closed)"
    return f"{doorentity.name}(open)"


def _format_location(location, options=None):
    if not location:
        return ''
    if options == 'kg-descr' or options == 'parsed-obs':
        return "unknown" if Location.is_unknown(location) else location.name

    return ":{}[{}]".format('', location.name)


class Connection:
    """
    A connection between two locations:

    from_location: The location that was departed
    action: The navigational action used
    to_location: The location arrived at, or None
    message: The text response given by the game upon moving

    """
    def __init__(self, from_location, action, to_location=None, doorway=None, message=''):
        self.from_location = from_location
        self.to_location   = to_location
        self.action        = action
        self.message       = message
        self.doorway       = doorway

    def __eq__(self, other):
        if isinstance(self, other.__class__):
            return self.action == other.action and\
                self.from_location == other.from_location and\
                self.to_location == other.to_location and\
                self.action == other.action and \
                self.doorway == other.doorway
        return False

    def update(self, other: 'Connection') -> 'Connection':
        updated = []
        if not Location.is_unknown(other.from_location):
            if self.from_location != other.from_location:
                assert Location.is_unknown(self.from_location), \
                    f"Connection endpoints aren't expected to change over time {self} <= {other}"
                updated.append("from_location")
                self.from_location = other.from_location
        if not Location.is_unknown(other.to_location):
            if self.to_location != other.to_location:
                assert Location.is_unknown(self.to_location), \
                    f"Connection destinations aren't expected to change over time {self} <= {other}"
                updated.append("to_location")
                self.to_location = other.to_location
        if other.action:
            if self.action != other.action:
                print(f"WARNING: replacing connection action {self} with {other.action}")
                updated.append("action")
            self.action = other.action
        if other.doorway:
            if self.doorway:
                assert self.doorway == other.doorway
            else:
                updated.append("doorway")
            self.doorway = other.doorway
        # if updated:
        #     print(f"ConnectionGraph updated {self} {updated} from {other}")
        return self

    def to_string(self, prefix='', options=None):
        if options == 'kg-descr':       # info known by current (non-GT) knowledge graph
            return prefix + "{}{} to {}".format(self.action.verb,
                    _format_doorinfo(self.doorway, options=options),
                    _format_location(self.to_location, options=options))
        elif options == 'parsed-obs':   # info directly discernible from 'look' command
            return prefix + "{}{} ;".format(self.action.verb,
                    _format_doorinfo(self.doorway, options=options))
        # else:
        return prefix + "{} --({}:{})--> {}".format(_format_location(self.from_location),
                                                   self.action.verb,
                                                   _format_doorinfo(self.doorway),
                                                   _format_location(self.to_location))

    def __str__(self):
        return self.to_string()

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return hash(self.to_string())
