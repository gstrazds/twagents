from typing import Iterable
from collections import defaultdict, OrderedDict
from textworld.logic import Proposition, Variable, State, Signature
from textworld.generator import World

OBSERVABLE_RELATIONS = ('at', 'in', 'on', 'chopped', 'roasted', 'baked', 'fried', 'sliced', 'peeled')
DIRECTION_RELATIONS = ('north_of', 'south_of', 'west_of', 'east_of')


def is_observable_relation(relname):
    return relname in OBSERVABLE_RELATIONS


def find_link_direction(world, src, dest):
    for room in world.rooms:
        if room.name == src.name:
            for e, r2 in room.exits.items():
                if r2.name == dest.name:
                    return e
    print("WARNING: find_link_direction failed!", src, dest)
    return None


def add_extra_door_facts(world, world_facts, local_facts=None, where_fact=None):
    # adds to world_facts additional facts about door directions
    # and also about exits and door directions relative to the player if local_facts is not None
    if local_facts is not None:
        assert where_fact is not None
    door_facts = world.state.facts_with_signature(Signature('link', ('r', 'd', 'r')))
    if where_fact:
        the_player = where_fact.arguments[0]
        player_location = where_fact.arguments[1]

    # for room in world.rooms:
    #     print(room)
    #     for e, p in room.exits.items():
    #         print('\t', e, p)

    for df in door_facts:
        assert len(df.arguments) == 3
        r0, door, r1 = df.arguments
        direction = find_link_direction(world, r1, r0)
        world_facts.append(Proposition("{}_of".format(direction), (door, r0)
                                       ))
        if local_facts is not None and r1 == player_location:
            local_facts.append(Proposition("{}_of".format(direction), (
                door,
                the_player)))
            # if world.state.is_fact(Proposition('free', (player_location, r1))):
            #     local_facts.append(Proposition("{}_of".format(direction), (
            #         the_player,
            #         Variable("exit_{}".format(direction), 'e'))))
            if world.state.is_fact(Proposition('closed', [door])):
                closed_fact = Proposition('closed', [door])
                if closed_fact not in local_facts:
                    local_facts.append(closed_fact)
            if world.state.is_fact(Proposition('locked', [door])):
                locked_fact = Proposition('locked', [door])
                if locked_fact not in local_facts:
                    local_facts.append(locked_fact)


def filter_observables(world_facts: Iterable[Proposition], verbose=False):
    fixups = defaultdict(set)
    if not world_facts:
        return None

    # print("WORLD FACTS:")
    for fact in world_facts:
        # print('\t', fact)
        for v in fact.arguments:
            #     print('\t\t{}:{}'.format(v.name, v.type))
            if not v.name:
                v_count = len(fixups[v.type])
                assert v not in fixups[v.type]
                if v.type == 'P' or v.type == 'I':
                    v.name = v.type
                    assert v_count == 0
                else:
                    v.name = '~{}_{}'.format(v.type, v_count)
                fixups[v.type].add(v)

    world = World.from_facts(world_facts)
    world_state = world.state

    if 'P' in world_state._vars_by_type:
        players = world_state.variables_of_type('P')
        assert len(players) == 1
    #     for p in players:
    #         player = p
    # else:
    #     player = None

    where_sig = Signature('at', ('P', 'r'))
    where_am_i = world_state.facts_with_signature(where_sig)
    assert len(where_am_i) == 1
    where_fact = list(where_am_i)[0]
    the_player = where_fact.arguments[0]
    player_location = where_fact.arguments[1]
    if verbose:
        print("WORLD FACTS:")
        for fact in world_facts:
            print('\t', fact)
            # print_fact(game, fact)

    if verbose:
        print("VARIABLES:")
        for v in world_state.variables:
            print('\t\t{}:{}'.format(v.name, v.type))

    print(where_fact, world.player_room)
    facts_in_scope = world.get_facts_in_scope()
    observable_facts = []
    for fact in facts_in_scope:
        # print("***\t\t", fact)
        if is_observable_relation(fact.name):
            if fact != where_fact:
                observable_facts.append(fact)
        else:
            pass

    for e in world.player_room.exits:
        if world.state.is_fact(Proposition('free', (world.player_room.exits[e], where_fact.arguments[1]))):
            observable_facts.append(Proposition("{}_of".format(e), (
                                                                    Variable("exit_{}".format(e), 'e'),
                                                                    the_player
                                                                    )))
    # REFACTORING: the following is now handled within add_extra_door_facts()
    #     else:  # probably a closed door
    #         door_facts = world.state.facts_with_signature(Signature('link', ('r', 'd', 'r')))
    #         for df in door_facts:
    #             if df.arguments[0] == player_location:
    #                 the_door = df.arguments[1]
    #                 observable_facts.append(Proposition("{}_of".format(e), (
    #                                                                 the_player,
    #                                                                 the_door
    #                                                                 )))
    #                 if world.state.is_fact(Proposition('closed', [the_door])):
    #                     observable_facts.append(Proposition('closed', [the_door]))

    add_extra_door_facts(world, world_facts, local_facts=observable_facts, where_fact=where_fact)

    if verbose:
        print("++WORLD FACTS++:")
        for fact in world_facts:
            print('\t', fact)
            # print_fact(game, fact)
    return observable_facts, player_location

def get_obj_infos(infos, entities, room_infos, gid):
    # select just the non-room objects (also excluding abstract objects like "slots", "ingredients", etc)
    obj_types = ['stove', 'oven', 'toaster', 'P', 'o', 'c', 's', 'f', 'd', 'meal']
    # P=player, o=object, c=container, s=support, f=food, d=door
    excluded_types = ['ingredient', 'RECIPE', 'slot', 'I', 'r']  # I=inventory, r=room
    obj_infos = list(filter(lambda a: a['type'] in obj_types, infos))
    notexcluded_infos = list(filter(lambda a: a['type'] not in excluded_types, infos))
    if (len(notexcluded_infos) != len(obj_infos)):
        s_ne = set()
        s_o = set()
        for i in notexcluded_infos:
            s_ne.add(i['type'])
        for i in obj_infos:
            s_o.add(i['type'])
        print("!!!!!", gid, "EXTRA types:", s_ne.difference(s_o))
        assert len(notexcluded_infos) > len(obj_types)
        assert len(s_ne) > len(s_o)

    out_list = OrderedDict()
    for o in obj_infos:
        if o['type'] == 'P':
            o['name'] = '<PLAYER>'  # instead of null
        o_name = o['name']
        if o_name in out_list:
            print("!!!!!", gid, "WARNING: DUPLICATE OBJ NAME", o_name, out_list[o_name]['id'], o['id'])
        out_list[o_name] = o
    # double check that we found everything in the entities list
    if entities:
        s_ent = set(entities)
        s_found = set(out_list.keys())
        s_found.update(room_infos.keys())
        if s_ent.intersection(s_found) != s_ent:
            print("!!!!!", gid, "WARNING -- DIDN'T RESOLVE ALL ENTITIES:")
            print("\tNOT FOUND:", s_ent.difference(s_found))
    return out_list


def _filter_unnamed_and_room_entities(e):
    return e.name and e.type != "r"


def _filter_rooms(e):
    return e.name and e.type == "r"


def find_room_info(game, room_name):
    room_infos = [ri for ri in filter(_filter_rooms, game.infos.values())
                    if ri.name == room_name or ri.id == room_name]
    if room_infos:
        return room_infos[0]
    else:
        print("------failed to find room info for '{}'".format(room_name))


def find_entity_info(game, entity_name):
    entity_infos = [ei for ei in filter(_filter_unnamed_and_room_entities,
                                       game.infos.values()) if ei.name == entity_name]
    return entity_infos[0] if entity_infos else None


def find_info(game, info_key):  # info_key might be info.id or info.name
    if info_key in game.infos:
        return game.infos[info_key]
    entity_infos = [ei for ei in game.infos.values()
                    if ei.name == info_key or ei.id == info_key]
    return entity_infos[0] if entity_infos else None


def print_variable(game, arg):
    info = None
    outname = ''
    if arg.type == 'r':
        info = find_room_info(game, arg.name)
        info_id = info.id
        outname = info.name
    else:
        info_id = arg.name
        if not info_id:
            info_id = arg.type
        if info_id:
            if info_id in game.infos:
                info = game.infos[info_id]
            else:
                info = find_entity_info(game, info_id)
            if info:
                info_id = info.id
                if arg.type == 'I':
                    outname = 'Inventory'
                elif info.type == 'P':
                    outname = 'Player'
                else:
                    outname = info.name
    print("'{}'[{}]".format(outname, info_id), end='')


def print_fact(game, fact):
    argrest = 0
    if len(fact.arguments) > 0:
        print_variable(game, fact.arguments[0])
        print(' ', end='')
        argrest = 1
    print(":{}:".format(fact.name), end='')
    if len(fact.arguments) > argrest:
        for arg in fact.arguments[argrest:]:
            print(' ', end='')
            print_variable(game, arg)
    print()


def format_adj(adj):
    return 'None' if adj is None else "'{}'".format(adj)
