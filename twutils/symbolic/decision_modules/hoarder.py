from ..decision_module import DecisionModule
from ..knowledge_graph import *
from ..action import *
from ..game import GameInstance
from .. import gv


class Hoarder(DecisionModule):
    """ The hoarder attempts to Take All """
    def __init__(self, active=False):
        super().__init__()
        self._active = active

    def process_event(self, event, gi: GameInstance):
        if not self._active:
            return
        # if type(event) is NewLocationEvent and gv.TakeAll.recognized():
        #     self._eagerness = 1.

    def parse_response(self, response, gi: GameInstance):
        here = gi.kg.player_location
        success = False
        for line in response.splitlines():
            line = line.strip()
            if ':' in line:
                success = True
                # Example - small mailbox: It's securly anchored.
                entity_name, resp = [w.strip() for w in line.split(':', 1)]
                short_name = entity_name.split(' ')[-1]
                if here.has_entity_with_name(entity_name):
                    entity = here.get_entity_by_name(entity_name)
                elif gi.kg.inventory.has_entity_with_name(entity_name):
                    entity = gi.kg.inventory.get_entity_by_name(entity_name)
                else:
                    # Create the entity at the current location
                    entity = Entity(entity_name, here)
                    entity.add_name(short_name)
                    gi.entity_at_location(entity, here)  # FORMERLY: here.add_entity(entity)

                take_action = gv.Take(entity)
                p_valid = take_action.validate(resp)
                gv.dbg("[Take] p={:.2f} {} --> {}".format(p_valid, entity_name, resp))
                gi.act_on_entity(take_action, entity, p_valid, resp)
                if p_valid > 0.5:
                    take_action.apply(gi)
        self.record(success)

    def take_control(self, gi: GameInstance):
        obs = yield
        response = yield gv.TakeAll
        self.parse_response(response, gi)
        self._eagerness = 0
