import re
from typing import Iterable, Mapping, List, Tuple, Dict
from collections import defaultdict, OrderedDict
from textworld.logic import Proposition, Variable, Signature, State
from textworld.generator import World
from textworld.generator.game import Game, EntityInfo

OBSERVABLE_RELATIONS = ('at', 'in', 'on',
                        'chopped', 'diced', 'sliced', 'peeled', 'cut',
                        'baked', 'fried', 'roasted', 'cooked', 'grilled',
                        'closed', 'open', 'locked')
DIRECTION_RELATIONS = ('north_of', 'south_of', 'west_of', 'east_of')

COOK_WITH = {
    "grill": "BBQ",
    "bake": "oven",
    "roast": "oven",
    "fry": "stove",
    "toast": "toaster",
}

CUT_WITH = {
    "chop": "knife",
    "dice": "knife",
    "slice": "knife",
    "mince": "knife",
}


_INVERSE_DIRECTION = {
    'north_of': 'south_of',
    'east_of': 'west_of',
    'south_of': 'north_of',
    'west_of': 'east_of',
}
def reverse_direction(direction):
    return _INVERSE_DIRECTION[direction]


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
    else:
        the_player, player_location = None, None   # redundant, suppress warnings

    # for room in world.rooms:
    #     print(room)
    #     for e, p in room.exits.items():
    #         print('\t', e, p)

    for df in door_facts:
        assert len(df.arguments) == 3
        r0, door, r1 = df.arguments
        direction = find_link_direction(world, r1, r0)
        new_fact = Proposition("{}_of".format(direction), (door, r1))
        world_facts.append(new_fact)
        if local_facts is not None and r1 == player_location:
            local_facts.append(new_fact)
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
            # locked state is not directly observable
            # if world.state.is_fact(Proposition('locked', [door])):
            #     locked_fact = Proposition('locked', [door])
            #     if locked_fact not in local_facts:
            #         local_facts.append(locked_fact)



def reconstitute_facts(facts_list):
    if not isinstance(facts_list[0], Proposition):   # maybe serialized list of facts
        # world_facts0 = world_facts.copy()
        facts_list = [Proposition.deserialize(fact_json) for fact_json in facts_list]
        # print("Deserialized:\n", facts_list, "\n===>\n", facts_list)
    return facts_list


def filter_observables(world_facts: Iterable[Proposition], verbose=False, game=None):
    fixups = defaultdict(set)
    if not world_facts:
        return None
    world_facts = reconstitute_facts(world_facts)

    # print("WORLD FACTS:")
    for fact in world_facts:
        # print('\t', fact)
        for v in fact.arguments:
            #     print('\t\t{}:{}'.format(v.name, v.type))
            if not v.name:
                assert False, f"fact.arguments are all expected to have a non-empty .name property {v} / {fact}"
                v_count = len(fixups[v.type])
                assert v not in fixups[v.type]
                if v.type == 'P' or v.type == 'I' or v.type == 'RECIPE' or v.type == 'MEAL':
                    v.name = v.type
                    if v_count == 0:
                        fixups[v.type].add(v)
                else:
                    v.name = '~{}_{}'.format(v.type, v_count)
                    fixups[v.type].add(v)
    # print("((((( filter_observables:")
    # for key in fixups.keys():
    #     print(f"\t{key}: {fixups[key]}")
    # print(")))))")
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
    # if verbose:
    #     print("WORLD FACTS:")
    #     for fact in world_facts:
    #         print('\t', fact)
    #         # print_fact(game, fact)

    # if verbose:
    #     print("VARIABLES:")
    #     for v in world_state.variables:
    #         print('\t\t{}:{}'.format(v.name, v.type))

    print(where_fact, world.player_room)
    facts_in_scope = world.get_facts_in_scope()
    observable_facts = []
    for fact in facts_in_scope:
        # print("***\t\t", fact)
        if is_observable_relation(fact.name):
            if fact != where_fact:
                observable_facts.append(fact)
            else:  # consider the player's current location to be directly observable
                observable_facts.append(fact)
        else:
            pass

    for e in world.player_room.exits:
        if world.state.is_fact(Proposition('free', (world.player_room.exits[e], where_fact.arguments[1]))):
            observable_facts.append(Proposition("{}_of".format(e), (
                                                                    Variable("exit_{}".format(e), 'e'),
                                                                    world.player_room   # the_player
                                                                    )))
    # REFACTORING: adding e.g. closed(door) and [north|east|west|south_of](P,door) now in add_extra_door_facts()
    add_extra_door_facts(world, world_facts, local_facts=observable_facts, where_fact=where_fact)

    if verbose:
        print("++WORLD FACTS++:")
        for fact in world_facts:
            prettyprint_fact(fact, game=game)

    return observable_facts, player_location


def remap_proposition(prop: Proposition, names2ids_map: Mapping[str,str]) -> Proposition:
    def _remap_var(arg: Variable, names2ids_map) -> Variable:
        _name = names2ids_map.get(arg.name, arg.name)
        return Variable(_name, arg.type)
    args_new = [_remap_var(arg, names2ids_map) for arg in prop.arguments]
    prop_new = Proposition(prop.name, args_new)
    print(f"remap_proposition: {prop} --> {prop_new}")
    return prop_new


def subst_names(obstxt:str, rename_map: Mapping[str,str]) -> str:
    sorted_list = list(sorted(rename_map.items(), key=lambda tup: len(tup[0]), reverse=True))
    edited_str = obstxt
    # replace longer names first
    print(sorted_list)
    for varname, varid in sorted_list:
        assert varname and varid, f"varname:{varname} varid:{varid} EXPECTED TO BE NONEMPTY!"
        # TODO: ?better handling of special objs [meal, rooms, etc]
        if varname != varid and not varid.startswith(varname+'_0'):    # don't substitute meal->meal_0, stove->stove_0, etc
            print(f"Replacing {varname}<-{varid}")
            edited_str = re.sub(varname, varid, edited_str, flags=re.IGNORECASE)   #TODO: could optimize by precompiling and remembering
        # print("\t-> ", edited_str)
    print("MODIFIED observation: >>>", edited_str)
    print("<<<----------end of MODIFIED observation")
    return edited_str


def remap_observation_and_facts(obstxt, observed_facts, rename_map: Mapping[str,str]):
    obs_facts = [remap_proposition(f, rename_map) for f in observed_facts]
    if obstxt:
        obstxt = subst_names(obstxt, rename_map)
    return obstxt, obs_facts


def remap_command(cmdstr: str, rename_map: Mapping[str,str]):
    if cmdstr:
        cmdstr = subst_names(cmdstr, rename_map)
    return cmdstr


def human_readable_to_internal(game:Game, fact:Proposition) -> Proposition:
    def _human_readable_to_internal_var(game:Game, arg:Variable) -> Variable:
        _name, _type = lookup_internal_info(game, arg)
        return Variable(_name, _type)

    args_new = [ _human_readable_to_internal_var(game, arg) for arg in fact.arguments ]
    fact_new = Proposition(fact.name, args_new)
    return fact_new


def prettyprint_fact(fact, game=None, indent=1):
    if indent > 0:
        print('\t' * indent, end='')
    if game:
        print_fact(game, fact)
    else:
        print(fact)


def get_obj_infos(infos, entities, room_infos, gid):
    # select just the non-room objects (also excluding abstract objects like "slots", "ingredients", etc)
    obj_types = ['stove', 'oven', 'toaster', 'P', 'o', 'c', 's', 'f', 'd', 'meal']
    # P=player, o=object, c=container, s=support, f=food, d=door
    excluded_types = ['ingredient', 'RECIPE', 'slot', 'I', 'r']  # I=inventory, r=room
    obj_infos = list(filter(lambda a: a['type'] in obj_types, infos))
    notexcluded_infos = list(filter(lambda a: a['type'] not in excluded_types, infos))
    if len(notexcluded_infos) != len(obj_infos):
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


def _filter_out_unnamed_and_room_entities(e):
    return e.name and e.type != "r"


def _select_rooms(e):
    return e.name and e.type == "r"


def find_room_info(game, room_name):
    room_name = room_name.lower()
    room_infos = [ri for ri in filter(_select_rooms, game.infos.values())
                    if ri.name.lower() == room_name or ri.id == room_name]
    if room_infos:
        return room_infos[0]
    else:
        # print("------failed to find room info for '{}'".format(room_name))
        return None

def find_entity_info(game, entity_name):
    #entity_name = entity_name.lower()
    entity_infos = [ei for ei in filter(_filter_out_unnamed_and_room_entities,
                                        game.infos.values()) if ei.name == entity_name]
    return entity_infos[0] if entity_infos else None


def _find_info(game, info_key):  # match EntityInfo.id or EntityInfo.name
    # info_key = info_key.lower()  # instance of RECIPE has no name, but has allcaps id = RECIPE
    if info_key in game.infos:
        return game.infos[info_key]
    entity_infos = [ei for ei in game.infos.values()
                    if ei.id == info_key or ei.name == info_key]
    return entity_infos[0] if entity_infos else None


def lookup_internal_info(game, arg) -> Tuple[str, str]:
    info_id = arg.name
    info = _find_info(game, info_id)
    if not info:
        print("lookup_internal_info failed for arg.name:", info_id)
        # info = _find_info(game, arg.name)
    if info:
        _name = info.id  # if info.id else info.type
        _type = info.type
    else:
        _name = info_id  # arg.name
        _type = arg.type
        if type and '_' in _type:
            print(f"!!!UNEXPECTED lookup_internal_info({info_id}) splitting pseudo arg.type:{_type}")
            _type = _type.split('_')[0]
    print(f"lookup_internal_info: {arg} -> name:{_name}, type:{_type}")
    return _name, _type


def lookup_internal_id(game, var_name) -> str:
    info = _find_info(game, var_name)
    _name_id = info.id  if info else None
    print(f"lookup_internal_id: {var_name} -> {_name_id}")
    return _name_id


def get_name2idmap(game) -> Dict[str,str]:
    names2ids:Dict[str,str] = {}
    for key, info in game.infos.items():
        if info.name and info.name != info.id and not info.id.startswith(info.name+'_0'):    # don't substitute meal->meal_0, stove->stove_0, etc
            names2ids[info.name] = info.id
    print("get_name2idmap: ", names2ids)
    return names2ids

def print_variable(game, arg):
    _name, _type_ = lookup_internal_info(game, arg)
    obj_id = arg.type

    if _name == 'I':
       outname = 'Inventory'
    elif _name == 'P':
        outname = 'Player'
    else:
        outname = _name
    print("'{}'[{}]".format(outname, obj_id), end='')


def print_fact(game, fact):
    argrest = 0
    if len(fact.arguments) > 0:
        print_variable(game, fact.arguments[0])
        print(' ', end='')
        argrest = 1
    print(":---{}".format(fact.name), end='')
    if len(fact.arguments) > argrest:
        if len(fact.arguments[argrest:]) > 1:
            print("{", end='')
            print_variable(game, fact.arguments[argrest])
            print("}", end='')
            argrest += 1
        print("---: ", end='')
        if len(fact.arguments[argrest:]) > 1:
            for arg in fact.arguments[argrest:-1]:
                print_variable(game, arg)
                print(', ', end='')
        print_variable(game, fact.arguments[-1])
    print()


def format_adj(adj):
    return 'None' if adj is None else "'{}'".format(adj)


def get_recipe(game, verbose=False):
    recipe = None
    # RECIPE_OID = 'o_0'  # HACK: In TW-cooking games, the first object is always the cookbook
    # if RECIPE_OID in game.infos and game.infos[RECIPE_OID].desc:
    #     cookbook_info = game.infos[RECIPE_OID]
    cookbook_info = _find_info(game, 'cookbook')
    if cookbook_info and cookbook_info.desc and cookbook_info.desc.find("Recipe #1") > -1:
        recipe = cookbook_info.desc
        if verbose:
            print("----------BEGIN-RECIPE---------")
            print(recipe)
            print("---------END-OF-RECIPE------")
    else:
        print("WARNING: get_recipe() - Recipe NOT FOUND")
    return recipe


def print_ids_and_names(game):
    entities_infos = filter(_filter_out_unnamed_and_room_entities, game.infos.values())
    room_infos = filter(_select_rooms, game.infos.values())
    print("ROOMS:")
    for info in room_infos:
        print("{}: '{}'".format(info.id, info.name))
    print("--------------")
    for info in entities_infos:  # game1.infos.values()
        print("{}: '{}' [{}]".format(info.id, info.name, format_adj(info.adj)))  # , info.type ))


def _convert_cooking_instruction(words, device: str, change_verb=None):
    words_out = words.copy()
    words_out.append("with")
    words_out.append(device)
    if change_verb:
        words_out[0] = change_verb  # convert the verb to generic "cook" (the specific verbs don't work as is in TextWorld)
    return words_out


def adapt_tw_instr(words: str, kg) -> Tuple[List[str],List[str]]:
    # if instr.startswith("chop ") or instr.startswith("dice ") or instr.startswith("slice "):
    #     return instr + " with the knife", ["knife"]
    # words = instr.split()
    with_objs = []
    if words[0] in COOK_WITH:
        device = COOK_WITH[words[0]]
        with_objs.append(device)
        return _convert_cooking_instruction(words, device, change_verb="cook"), with_objs
    elif words[0] in CUT_WITH:
        device = CUT_WITH[words[0]]
        with_objs.append(device)
        return _convert_cooking_instruction(words, device), with_objs
    else:
        return words, []

# adapted from QAit (customized) version of textworld.generator.game.py
def game_objects(game: Game) -> List[EntityInfo]:
    """ The entity information of all relevant objects in this game. """
    def _filter_unnamed_and_room_entities(e):
        return e.name and e.type != "r"
    return filter(_filter_unnamed_and_room_entities, game.infos.values())


def game_object_names(game: Game) -> List[str]:
    """ The names of all objects in this game. """
    get_names_method = getattr(game, "object_names", None)  # game from TextWorld-QAit (customized)
    if get_names_method and callable(get_names_method):
        print(f"game_object_names<TextWorld-QAit>({game.metadata['uuid']}")
        return game.get_names_method()
    get_names_method = getattr(game, "objects_names", None)  # game from TextWorld (1.3.x)
    if get_names_method and callable(get_names_method):
        print(f"game_object_names<TextWorld-1.3.x>({game.metadata['uuid']}")
        return game.get_names_method()
    print(f"game_object_names<local -- game_objects()>({game.metadata['uuid']}")
    return [entity.name for entity in game_objects(game)]


def game_object_nouns(game: Game) -> List[str]:
    """ The noun parts of all objects in this game. """
    _object_nouns = [entity.noun for entity in game_objects(game)]
    return _object_nouns


def game_object_adjs(game: Game) -> List[str]:
    """ The adjective parts of all objects in this game. """
    _object_adjs = [entity.adj if entity.adj is not None else '' for entity in game_objects(game)]
    return _object_adjs


# from challenge.py (invoked during game generation)
# Collect infos about this game.
# metadata = {
#     "seeds": options.seeds,
#     "settings": settings,
# }
#
# def _get_fqn(obj):
#     """ Get fully qualified name """
#     obj = obj.parent
#     name = ""
#     while obj:
#         obj_name = obj.name
#         if obj_name is None:
#             obj_name = "inventory"
#         name = obj_name + "." + name
#         obj = obj.parent
#     return name.rstrip(".")
#
# object_locations = {}
# object_attributes = {}
# for name in game.object_names:
#     entity = M.find_by_name(name)
#     if entity:
#         if entity.type != "d":
#             object_locations[name] = _get_fqn(entity)
#         object_attributes[name] = [fact.name for fact in entity.properties]
#
# game.extras["object_locations"] = object_locations
# game.extras["object_attributes"] = object_attributes

def parse_ftwc_recipe(recipe_str:str, format='fulltext'):
    # parse the observation string (from a successful 'read cookbook' action)
    recipe_lines = recipe_str.split('\n')
    recipe_lines = list(map(lambda line: line.strip(), recipe_lines))
    recipe_lines = list(map(lambda line: line.lower(), recipe_lines))
    ingredients = []
    directions = []
    has_ingredients = False
    start_of_ingredients = 0
    start_of_directions: int = len(recipe_lines) + 1000   # default: past the end of data
    # try:
    #     start_of_ingredients = recipe_lines.index("Ingredients:")
    #     start_of_ingredients += 1
    # except ValueError:
    #     print("RecipeReader failed to find Ingredients in:", recipe_str)
    #     start_of_ingredients = 0
    for i, line in enumerate(recipe_lines[start_of_ingredients:]):
        if line.startswith("ingredients"):
            has_ingredients = True
            start_of_ingredients = i + 1
            break
    for i, ingredient in enumerate(recipe_lines[start_of_ingredients:]):
        if ingredient.startswith("directions"):
            start_of_directions = start_of_ingredients + i + 1
            break  # end of Ingredients list
        if has_ingredients and ingredient:
            ingredients.append(ingredient)

    if start_of_directions < len(recipe_lines):
        # assert recipe_lines[start_of_directions - 1] == 'Directions:'
        for recipe_step in recipe_lines[start_of_directions:]:
            if recipe_step:
                if ' the ' in recipe_step:
                    recipe_step = recipe_step.replace(' the ', ' ')
                directions.append(recipe_step)
    return ingredients, directions
