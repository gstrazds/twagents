from ..decision_module import DecisionModule
from ..action import Take
from ..event import NeedToAcquire, NeedToFind, NeedToGoTo, NoLongerNeed
from ..game import GameInstance
from .. import gv


class GTAcquire(DecisionModule):
    """ The Ground Truth Acquire module locates and, for takeable objects, takes specific objects (into inventory) """
    def __init__(self, active=False):
        super().__init__()
        self.required_objs = set()
        self.found_objs = set()
        self._active = active

    def add_required_obj(self, entityname:str):
        print("GTAcquire.add_required_obj({})".format(entityname))
        self.required_objs.add(entityname)

    def remove_required_obj(self, entityname:str):
        self.required_objs.discard(entityname)
        self.found_objs.discard(entityname)

    def missing_objs(self, kg):
        if not self.required_objs:
            return False
        for name in self.required_objs:
            e = kg.inventory.get_entity_by_name(name)
            if e:
                self.found_objs.add(name)
        return self.required_objs - self.found_objs

    def process_event(self, event, gi: GameInstance):
        # if not self._active:
        #     return
        # if type(event) is NewLocationEvent and gv.TakeAll.recognized():
        #     self._eagerness = 1.
        if event.is_groundtruth:
            if isinstance(event, NeedToAcquire) or isinstance(event, NeedToFind):
                print("GT Acquire:", event.objnames)
                for itemname in event.objnames:
                    self.add_required_obj(itemname)
                    self._eagerness = 0.8

    def parse_response(self, response, gi: GameInstance):
        success = False
        # here = gi.kg.player_location
        # for line in response.splitlines():
        #     line = line.strip()
        #     if ':' in line:
        #         success = True
        #         # Example - small mailbox: It's securly anchored.
        #         entity_name, resp = [w.strip() for w in line.split(':', 1)]
        #         short_name = entity_name.split(' ')[-1]
        #         if here.has_entity_with_name(entity_name):
        #             entity = here.get_entity_by_name(entity_name)
        #         elif gi.kg.inventory.has_entity_with_name(entity_name):
        #             entity = gi.kg.inventory.get_entity_by_name(entity_name)
        #         else:
        #             # Create the entity at the current location
        #             entity = Entity(entity_name, here)
        #             entity.add_name(short_name)
        #             gi.entity_at_location(entity, here)  # FORMERLY: here.add_entity(entity)
        #
        #         take_action = gv.Take(entity)
        #         p_valid = take_action.validate(resp)
        #         gv.dbg("[Take] p={:.2f} {} --> {}".format(p_valid, entity_name, resp))
        #         gi.act_on_entity(take_action, entity, p_valid, resp)
        #         if p_valid > 0.5:
        #             take_action.apply(gi)
        self.record(success)

    def take_control(self, gi: GameInstance):
        ignored_objs = yield
        still_needed = list(self.missing_objs(gi.gt))
        print("GT_Acquire required_objs:", self.required_objs)
        print("GT_Acquire still_needed:", still_needed)
        here = gi.gt.player_location
        for entityName in still_needed:
            if gi.gt.inventory.has_entity_with_name(entityName):
                assert False, f"Still needed object {entityName} *should not be* in Inventory"
            elif here.has_entity_with_name(entityName):
                entity = here.get_entity_by_name(entityName)
                if 'portable' in entity.attributes:
                    take_action = Take(entity)
                    response = yield take_action
                else: # can't take it, but we've found it, so consider it acquired...
                    self.found_objs.add(entityName)

        still_needed = list(self.missing_objs(gi.gt))
        if still_needed:
            obj_to_find = still_needed[0]   # TODO: make it smarter - choose closest instead of first
            locs = gi.gt.location_of_entity(obj_to_find)
            if locs:
                loc = list(locs)[0]
                print("GT_Acquire NeedToGoTo:", loc)
                gi.event_stream.push(NeedToGoTo(loc.name, groundtruth=True))
            else:
                print(f"Don't know GT location of {obj_to_find}")
                # assert False, f"Don't know GT location of {obj_to_find}"
                self.remove_required_obj(obj_to_find)
                gi.event_stream.push(NoLongerNeed([obj_to_find], groundtruth=True))
                self._eagerness = 0
        else:
            self._eagerness = 0
        return None  # terminate generator iteration

