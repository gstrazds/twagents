from typing import Iterable, List
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


def reverse_direction(direction):
    _DIRECTIONS_MAP = {
        'north_of': 'south_of',
        'east_of': 'west_of',
        'south_of': 'north_of',
        'west_of': 'east_of',
    }
    return _DIRECTIONS_MAP[direction]


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


def filter_observables(world_facts: List[Proposition], verbose=False, game=None):
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
                v_count = len(fixups[v.type])
                assert v not in fixups[v.type]
                if v.type == 'P' or v.type == 'I' or v.type == 'RECIPE' or v.type == 'MEAL':
                    v.name = v.type
                    if v_count == 0:
                        fixups[v.type].add(v)
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
            prettyprint_fact(fact, game=game)
    return observable_facts, player_location


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
    entity_name = entity_name.lower()
    entity_infos = [ei for ei in filter(_filter_out_unnamed_and_room_entities,
                                        game.infos.values()) if ei.name == entity_name]
    return entity_infos[0] if entity_infos else None


def find_info(game, info_key):  # info_key might be info.id or info.name
    info_key = info_key.lower()
    if info_key in game.infos:
        return game.infos[info_key]
    entity_infos = [ei for ei in game.infos.values()
                    if ei.name == info_key or ei.id == info_key]
    return entity_infos[0] if entity_infos else None


def print_variable(game, arg):
    info = None
    outname = ''
    if False and arg.type == 'r':
        info = find_room_info(game, arg.name)
    else:
        info_id = arg.name
        if not info_id:
            info_id = arg.type
        if info_id:
            if info_id in game.infos:
                info = game.infos[info_id]
            else:
                info = find_info(game, info_id)
    if info:
        info_id = info.id
        if arg.type == 'I':
            outname = 'Inventory'
        elif info.type == 'P':
            outname = 'Player'
        else:
            outname = info.name
    else:
        outname = arg.name
        info_id = arg.type
        # print("ARG:", arg)

    print("'{}'[{}]".format(outname, info_id), end='')


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


def get_recipe(game, verbose=False):   # HACK: In TW-cooking games, the first object is always the recipe
    RECIPE_OID = 'o_0'
    recipe = None
    if RECIPE_OID in game.infos and game.infos[RECIPE_OID].desc:
        if game.infos[RECIPE_OID].desc.find("Recipe #1") > -1:
            recipe = game.infos[RECIPE_OID].desc
    if verbose:
        print("----------BEGIN-RECIPE---------")
        print(recipe)
        print("---------END-OF-RECIPE------")
    else:
        print("Recipe NOT FOUND")
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


def adapt_tw_instr(words: str, kg) -> str:
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

def simplify_feedback(feedback_str: str):
    if not feedback_str:
        return ''
    feedback_str = feedback_str.strip()
    if "cook a delicious meal" in feedback_str and "cookbook in the kitchen" in feedback_str:
        feedback_str = "You should : find kitchen , read cookbook , eat meal ."
    elif feedback_str.endswith(" and look around"):  # this is a preprocessed feedback msg from QaitGymEnvWrapper
        feedback_str = feedback_str[:-16]+"."   # remove " and look around" (because it's redundant)
    elif feedback_str.endswith(" and look around"):  # this is a preprocessed feedback msg from QaitGymEnvWrapper
        feedback_str = feedback_str[:-16]+"."   # remove " and look around" (because it's redundant)
    elif "all following ingredients and follow the directions to prepare" in feedback_str:
        ingredients, directions = parse_ftwc_recipe(feedback_str)
        if ingredients or directions:
            feedback_str = "You read the recipe ---------"
            if ingredients:
                feedback_str += f" Acquire : " + " , ".join(ingredients) + " ;"
            if directions:
                feedback_str += f" Do : " + " , ".join(directions) + " ;"
    elif "our score has" in feedback_str:  # strip out useless lines about 'score has gone up by one point"
        feedback_lines = feedback_str.split('\n')
        output_lines = []
        for line in feedback_lines:
            line = line.strip()
            if "score has " in line:
                continue
            if "ou scored " in line:  # You scored x out of a possible y ...
                continue
            if "*** the end ***" in line.lower():
                continue
            if "would you like to quit" in line.lower():
                continue
            if line:
                if line.endswith(" Not bad."):  # You eat the meal. Not bad.
                    line = line[0:-9]
                elif "dding the meal to your " in line:  # Adding the meal to your inventory.
                    line = "You prepare the meal."
                output_lines.append(line)
        feedback_str = "\n".join(output_lines)
    return feedback_str