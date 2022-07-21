from typing import Tuple
import re    # regular expressions, for finding mentions of entities

from symbolic.action import *
from symbolic.entity import ConnectionRelation, Entity, Thing, Location, Person, UnknownLocation, Door
from symbolic.entity import DOOR, ROOM, PERSON, INVENTORY, CONTAINED_LOCATION, SUPPORTED_LOCATION, NONEXISTENCE_LOC


def _span_len(span: Tuple):
    return span[1] - span[0]

def _span_overlap(span1: Tuple, span2: Tuple):
    if span1[1] <= span2[0] or span1[0] >= span2[1]:
        return 0  # NO OVERLAP
    if span1[0] >= span2[0] and span1[0] < span2[1]:  # they overlap!
        return min(span2[1], span1[1]) - span1[0]
    return _span_overlap(span2, span1)   # same checks, other ordering

def test_span_overlap():
    overlaps = [
        _span_overlap((0, 1), (0, 1)),    # identical len 1   => 1
        _span_overlap((10, 11), (10, 11)), # identical len 1   => 1
        _span_overlap((2, 4), (2, 4)),     # identical 2 pos   => 2
        _span_overlap((5, 8), (6, 20)),    # partial overlap   => 2
        _span_overlap((6, 20), (5, 8)),    # -> 2
        _span_overlap((5, 10), (4, 10)),   # common end        => 5
        _span_overlap((4, 10), (5, 10)),   # command end       => 5
        _span_overlap((5, 10), (5, 11)),   # common start      => 5
        _span_overlap((5, 11), (5, 10)),   # common start      => 5
    ]
    if overlaps != [1,1,2,2,2,5,5,5,5]:
        assert overlaps == [1, 1, 2, 2, 2, 5, 5, 5, 5]
        return False

    non_overlaps = [
        _span_overlap((5, 7), (8, 10)),  # disjoint
        _span_overlap((8, 10), (5, 7)),  # disjoint
        _span_overlap((0, 1), (1, 2)),  # adjacent 1 pos
        _span_overlap((1, 2), (0, 1)),  # adjacent 1 pos
        _span_overlap((10, 11), (11, 12)),  # adjacent 1 pos
        _span_overlap((11, 12), (12, 11)),  # adjacent 1 pos
        _span_overlap((10, 15), (15, 20)),  # adjacent longer
        _span_overlap((15, 20), (10, 15)),  # adjacent longer
    ]
    _zero_overlaps = [0] * 8
    if non_overlaps != _zero_overlaps:
        assert non_overlaps == _zero_overlaps
        return False
    return True

test_span_overlap()   # module won't load if this test fails!!!


class MentionIndex:
    def __init__(self, observ_str:str):
        self._observ_str = observ_str
        self.mentions = {}  # dict like {name: List[Tuple(start,end)]}

    def is_subspan_of_existing(self, span, name=''):
        _my_len = _span_len(span)
        for k in self.mentions:
            span_list = self.mentions[k]
            for span0 in span_list:
                overlap = _span_overlap(span0, span)
                if overlap:
                    if _my_len == overlap:   # full inclusion
                        print(f"New span for [{name}]{span} is included in [{k}]{span0}")
                        return True
                    if _my_len <= _span_len(span0):
                        if span[0] < span0[0] or span[1] > span0[1]:
                            print(f"!!! WARNING: ignoring SHORTER new span for [{name}]{span} OVERLAPS [{k}]{span0}")
                        return True
                    else:  # _my_len > _span_len(span0):
                        #print(f"WARNING!!! LONGER new span for [{name}]{span} OVERLAPS [{k}]{span0} => will unindex [{k}]{span0}")
                        pass
        return False

    def remove_subspans_of(self, span, name=None):
        """ if any existing spans are (shorter) subspans of span, remove them from the index"""
        _my_len = _span_len(span)
        for k in self.mentions:
            span_list = self.mentions[k]
            for span0 in list(span_list):  # iterate through a copy of the list
                overlap = _span_overlap(span0, span)
                if overlap:
                    if _my_len <= _span_len(span0):  # full inclusion or new is shorter
                        assert False, f"SHOULD ALREADY BE FILTERED OUT: shorter new span [{name}]{span} OVERLAPS [{k}]{span0}"
                        continue
                    else:  # _my_len > _span_len(span0):
                        print(f"UNINDEXING [{k}]{span0} because of LONGER new span: [{name}]{span}")
                        span_list.remove(span0)

    def get_first_mention(self, name: str) -> int:
        if name in self.mentions:
            return self.mentions[name][0][0]   # start position of first mention
        return -1   # mention not found

    def add_mention(self, name: str, span: Tuple):
        """ Adds a mention to the index.
         In case of an overlap with other mentions
          (which should be either a full inclusion or a superset),
         the shorter span is removed from the index and the longer is kept"""
        # print(f"add_mention({name}, {span}")
        if self.is_subspan_of_existing(span, name=name):
            return None
        else:
            self.remove_subspans_of(span, name)
            if name in self.mentions:
                if span not in self.mentions[name]:
                    self.mentions[name].append(span)
                    self.mentions[name] = sorted(self.mentions[name])  # maintain sorted ordering (short lists)
            else:
                self.mentions[name] = [span]
        return self.mentions[name][0]  # return span of first mention

    def index_mentions(self, name: str):  # adds all mentions
        # print("INDEX MENTIONS of:", name)
        matches = re.finditer(name, self._observ_str)
        for span in [m.span() for m in matches]:
            # print(name, ":", span)
            self.add_mention(name, span)
        # print(self.mentions)

    def sort_facts_by_first_mention(self, facts_list):   # returns sorted facts_list
        zip_list = []
        for fact in facts_list:
            pass
        return facts_list

    def sort_entity_list_by_first_mention(self, entities):   # returns sorted list of entity names (or entities)
        zip_list = []
        # print("PRE: sort_entity_list:", self.mentions)
        for entity in entities:
            if isinstance(entity, Entity):
                name = entity.name
            else:    # we can also sort name strings
                name = str(entity)

            mention_start = self.get_first_mention(name)
            if mention_start >= 0:
                zip_list.append((mention_start, entity))
            else:
                if isinstance(entity, Entity):
                    print("WARNING: SORT ENTITY LIST BY MENTION DIDN'T FIND", entity, entity.parent, entity.location)
                    # assert False
                else:
                    print("WARNING: SORT ENTITY LIST BY MENTION DIDN'T FIND str:", entity)
                zip_list.append((100000+len(zip_list), entity))
        # print("POST: sort_entity_list:", self.mentions)
        zip_list = sorted(zip_list)
        sorted_list = [entity for (start, entity) in zip_list]
        return sorted_list


END_OF_LIST = ';'
START_OF_LIST = ':'
ITEM_SEPARATOR = ','
LINE_SEPARATOR = '\n'


# NOTE: Here's a stove. [...] But the thing is empty. [misleading] (no clue here that stove is a support, not a container)
# on fridge <:> nothing <;>  in oven <:> nothing <;> on table <:> cookbook <;>  on counter <:> red potato <;> on stove: nothing.
# You carry: nothing.

#   # on floor: nothing. )
def describe_visible_objects(loc, inventory_items, exits_descr=None, mention_tracker=None, groundtruth=False):
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
        descr_str, idx = _format_entities_descr(non_door_entities, idx, groundtruth=groundtruth)
        obs_descr += descr_str + LINE_SEPARATOR
    if exits_descr:
        obs_descr += LINE_SEPARATOR + exits_descr
    if on_floor_entities:
        onground_descr = f"ON floor {START_OF_LIST} " + f" {ITEM_SEPARATOR} ".join([_format_entity_name(item) for item in on_floor_entities])
        obs_descr += LINE_SEPARATOR + onground_descr + ' '+END_OF_LIST
    if len(inventory_items):
        inventory_descr = f"Carrying {START_OF_LIST} " + f" {ITEM_SEPARATOR} ".join([_format_entity_name(item) for item in inventory_items])
        obs_descr += LINE_SEPARATOR + inventory_descr + ' '+END_OF_LIST
    return obs_descr


def _format_entity_name(entity):
    out_str = ""
    if entity.state:
        state_descr = entity.state.format_descr()
        if state_descr:
            out_str += state_descr
    if entity._type:
        out_str += f"_{entity._type}_ "
    out_str += f"{entity.name}"
    return out_str


def _format_entities_descr(entity_list, idx, groundtruth=False):
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
        and (not entity.has_been_opened or self._formatting_options == 'parsed-obs'):
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


