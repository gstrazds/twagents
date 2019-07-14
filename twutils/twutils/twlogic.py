from typing import Iterable
from collections import defaultdict
from textworld.logic import Proposition, Variable, State, Signature


def filter_observables(facts: Iterable[Proposition]):
    fixups = defaultdict(set)
    if facts:
        # print("WORLD FACTS:")
        for fact in facts:
            # print('\t', fact)
            for v in fact.arguments:
                #     print('\t\t{}:{}'.format(v.name, v.type))
                if not v.name:
                    v_count = len(fixups[v.type])
                    assert v not in fixups[v.type]
                    v.name = '~{}_{}'.format(v.type, v_count)
                    fixups[v.type].add(v)
        world_state = State(facts)
        # if 'P' in world_state._vars_by_type:
        #     players = world_state.variables_of_type('P')
        #     assert len(players) == 1
        #     for p in players:
        #         player = p
        # else:
        #     player = None

        print("WORLD FACTS:")
        for fact in facts:
            print('\t', fact)
        print("VARIABLES:")
        # player_found = False
        for v in world_state.variables:
            print('\t\t{}:{}'.format(v.name, v.type))
            # if player and v == player:
            #     player_found = True
        # print("Found Player:", player_found)
        where_sig = Signature('at', ('P', 'r'))
        where_am_i = world_state.facts_with_signature(where_sig)
        assert len(where_am_i) == 1
        where_fact = list(where_am_i)[0]
        print(where_fact)

    return None
