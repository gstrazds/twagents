from typing import List, Tuple, Any

from .action import Portable
from .entity import DOOR, ROOM, PERSON, INVENTORY, CONTAINER, SUPPORT, CONTAINED_LOCATION, SUPPORTED_LOCATION, NONEXISTENCE_LOC

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

    include_entity_type = (INCLUDE_ENTITY_TYPES and options != 'parsed-obs')
    idx = 0
    obs_descr = ""   #"You see:"+LINE_SEPARATOR
    while idx < len(non_door_entities):
        descr_str, idx = format_entities_descr(non_door_entities, idx, groundtruth=groundtruth, options=options)
        obs_descr += descr_str + LINE_SEPARATOR
    if exits_descr:
        obs_descr += LINE_SEPARATOR + exits_descr
    if on_floor_entities:
        onground_descr = f" {ITEM_SEPARATOR} ".join([_format_entity_name(item, include_entity_type) for item in on_floor_entities])
    else:
        onground_descr = "nothing" if (ALWAYS_LIST_ONGROUND and options != 'parsed-obs') else ''
    if onground_descr:
        obs_descr += LINE_SEPARATOR + f"ON floor {START_OF_LIST} {onground_descr} {END_OF_LIST}"
    if len(inventory_items):
        inventory_descr = f"Carrying {START_OF_LIST} " + f" {ITEM_SEPARATOR} ".join(
            [_format_entity_name(item, include_entity_type) for item in inventory_items])
    else:
        inventory_descr = f"Carrying {START_OF_LIST} nothing" if (ALWAYS_LIST_INVENTORY and options != 'parsed-obs') else ''
    if inventory_descr:
        obs_descr += LINE_SEPARATOR + inventory_descr + ' ' + END_OF_LIST
    return obs_descr


def _format_entity_name(entity, include_entity_type):
    out_str = ""
    if entity.state:
        state_descr = entity.state.format_descr()
        if state_descr:
            out_str += state_descr
    if include_entity_type and entity._type:
        out_str += f"_{entity._type}_ "
    out_str += f"{entity.name}"
    return out_str


def format_entities_descr(entity_list, idx, groundtruth=False, options=None):
    entity = entity_list[idx]
    idx += 1
    include_entity_type = (INCLUDE_ENTITY_TYPES and options != 'parsed-obs')
    if entity.is_container:
        descr_str = f"IN {_format_entity_name(entity, include_entity_type)} {START_OF_LIST} "
        held_entities = entity.entities
    elif entity.is_support:
        descr_str = f"ON {_format_entity_name(entity, include_entity_type)} {START_OF_LIST} "
        held_entities = entity.entities
    else:
        descr_str = f"{_format_entity_name(entity, include_entity_type)}"
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
            descr_str += _format_entity_name(entity_list[idx], include_entity_type)
            idx += 1
        descr_str += f" {END_OF_LIST}"
    return descr_str, idx


def format_exits(exits_list:List[str], options:str):
    if not exits_list:   # None or empty list
        if options == 'parsed-obs' or not ALWAYS_LIST_EXITS:
            return ""
        else:  # if there are no exits,
            exits_list = ["none"]
    return "Exits : " + f" {ITEM_SEPARATOR} ".join(exits_list) + f" {END_OF_LIST}{LINE_SEPARATOR}"
