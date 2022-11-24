#!/usr/bin/env python

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.


import sys
import argparse
from typing import Optional, Any

from textworld.generator import Game, KnowledgeBase, GameOptions
from textworld import GameMaker

def rewrite_game(game, options:Optional[GameOptions]):
 
    # Load knowledge base specific to this challenge.
    settings = game.metadata.get('settings', None)
    print(settings)
    if not settings:
        print("UNKNOWN SETTINGS")
        # print(game.metadata)
        if 'uuid' in game.metadata and '+drop+' in game.metadata['uuid']:
            print(f"DROP : {game.metadata['uuid']}")
        else:
            print(f"NO DROP : {game.metadata['uuid']}")
    elif settings.get("drop"):
        print("DROP")
        # options.kb = KnowledgeBase.load(logic_path=KB_LOGIC_DROP_PATH, grammar_path=KB_GRAMMAR_PATH)
    else:
        print("NO DROP")
        # options.kb = KnowledgeBase.load(logic_path=KB_LOGIC_PATH, grammar_path=KB_GRAMMAR_PATH)

    if not options or not options.path:
        print(f"{options}")
        # assert options.path, f"{options}"

    # rngs = options.rngs
    # rng_map = rngs['map']
    # rng_objects = rngs['objects']
    # rng_grammar = rngs['grammar']
    # rng_quest = rngs['quest']
    # rng_recipe = np.random.RandomState(settings["recipe_seed"])

    # M = GameMaker(options)

    # ...

    # M.grammar = textworld.generator.make_grammar(options.grammar, rng=rng_grammar, kb=options.kb)

    # M.quests = quests

    # G = compute_graph(M)  # Needed by the move(...) function called below.

    # Build walkthrough.
    # current_room = start_room

    # cookbook_desc = "You open the copy of 'Cooking: A Modern Approach (3rd Ed.)' and start reading:\n"
    # recipe = textwrap.dedent(
    #     """
    #     Recipe #1
    #     ---------
    #     Gather all following ingredients and follow the directions to prepare this tasty meal.

    #     Ingredients:
    #     {ingredients}

    #     Directions:
    #     {directions}
    #     """
    # )
    # recipe_ingredients = "\n  ".join(ingredient[0].name for ingredient in ingredients)

    # recipe_directions = []
    # for ingredient in ingredients:
    #     cutting_verb = TYPES_OF_CUTTING_VERBS.get(ingredient[2])
    #     if cutting_verb:
    #         recipe_directions.append(cutting_verb + " the " + ingredient[0].name)

    #     cooking_verb = TYPES_OF_COOKING_VERBS.get(ingredient[1])
    #     if cooking_verb:
    #         recipe_directions.append(cooking_verb + " the " + ingredient[0].name)

    # recipe_directions.append("prepare meal")
    # recipe_directions = "\n  ".join(recipe_directions)
    # recipe = recipe.format(ingredients=recipe_ingredients, directions=recipe_directions)
    # cookbook.infos.desc = cookbook_desc + recipe

    # game = M.build()

    # Collect infos about this game.
    # metadata = {
    #     "seeds": options.seeds,
    #     "goal": cookbook.infos.desc,
    #     "recipe": recipe,
    #     "ingredients": [(food.name, cooking, cutting) for food, cooking, cutting in ingredients],
    #     "settings": settings,
    #     "entities": [e.name for e in M._entities.values() if e.name],
    #     "nb_distractors": nb_distractors,
    #     "walkthrough": walkthrough,
    #     "max_score": sum(quest.reward for quest in game.quests),
    # }

    # objective = ("You are hungry! Let's cook a delicious meal. Check the cookbook"
    #              " in the kitchen for the recipe. Once done, enjoy your meal!")
    # game.objective = objective

    # game.metadata = metadata
    # skills_uuid = "+".join("{}{}".format(k, "" if settings[k] is True else settings[k])
    #                        for k in SKILLS if k in settings and settings[k])
    # uuid = "tw-cooking{split}-{specs}-{seeds}"
    # uuid = uuid.format(split="-{}".format(settings["split"]) if settings.get("split") else "",
    #                    specs=skills_uuid,
    #                    seeds=encode_seeds([options.seeds[k] for k in sorted(options.seeds)]))
    # game.metadata["uuid"] = uuid
    return game


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert games to use a different grammar, or to ASP code")
    parser.add_argument("games", metavar="game", nargs="+",
                        help="JSON files containing infos about a game.")
    parser.add_argument("--simplify-grammar", action="store_true",
                        help="Regenerate tw-cooking games with simpler text grammar.")

    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Verbose mode.")
    args = parser.parse_args()

    objectives = {}
    names = set()
    for game_filename in args.games:
        try:
            game_filename = game_filename.replace(".ulx", ".json")
            game_filename = game_filename.replace(".z8", ".json")
            game = Game.load(game_filename)
        except Exception as e:
            print("Cannot load {}.".format(game))
            if args.verbose:
                print(e)
            continue
        if args.simplify_grammar:
            new_game = rewrite_game(game, None)
            print(f"NEW GAME (KB): {new_game.kb}") # keys() = version, world, grammar, quests, infos, KB, metadata, objective 
            print(f"NEW GAME (grammar): {new_game.grammar}") # keys() = version, world, grammar, quests, infos, KB, metadata, objective 
        if args.verbose:
            print("ORIG GAME:", game.serialize())


