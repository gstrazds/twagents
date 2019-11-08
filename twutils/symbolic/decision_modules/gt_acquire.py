import random
from ..decision_module import DecisionModule
from ..action import Take, Drop, Open, Portable
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
        self._temporary_drop = None  # remember an object that we need to re-acquire
        self._no_longer_needed = set()

    def add_required_obj(self, entityname:str):
        print("GTAcquire.add_required_obj({})".format(entityname))
        self.required_objs.add(entityname)
        self._eagerness = 0.8

    def remove_required_obj(self, entityname:str):
        self.required_objs.discard(entityname)
        self.found_objs.discard(entityname)
        if not self.required_objs:
            self._eagerness = 0.0

    def missing_objs(self, kg):
        if not self.required_objs:
            return set()
        for name in self.required_objs:
            e = kg.inventory.get_entity_by_name(name)
            if e:
                self.found_objs.add(name)
        print(f"GTAcquire required:{self.required_objs} - found:{self.found_objs}")
        return self.required_objs - self.found_objs

    def process_event(self, event, gi: GameInstance):
        # if not self._active:
        #     return
        # if type(event) is NewLocationEvent and gv.TakeAll.recognized():
        #     self._eagerness = 1.
        if event.is_groundtruth:
            if isinstance(event, NeedToAcquire) or isinstance(event, NeedToFind):
                # print("GT Acquire:", event.objnames)
                for itemname in event.objnames:
                    self.add_required_obj(itemname)
                    # self._eagerness = 0.8
            elif isinstance(event, NoLongerNeed):
                print("GT Acquire NO LONGER NEED:", event.objnames)
                for itemname in event.objnames:
                    if itemname != self._temporary_drop:
                        self._no_longer_needed.add(itemname)
                    self.remove_required_obj(itemname)

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
        ignored_obs = yield
        if self._temporary_drop:
            gi.event_stream.push(NeedToAcquire([self._temporary_drop], groundtruth=True))
            self._temporary_drop = None
            yield None  # reevaluate which should be the active module

        still_needed = list(self.missing_objs(gi.gt))
        print("GT_Acquire required_objs:", self.required_objs)
        print("GT_Acquire still_needed:", still_needed)
        here = gi.gt.player_location
        for entityName in still_needed:
            if gi.gt.inventory.has_entity_with_name(entityName):
                assert False, f"Still needed object {entityName} *should not be* in Inventory"
            elif here.has_entity_with_name(entityName):
                entity = here.get_entity_by_name(entityName)
                if Portable in entity.attributes:
                    container = gi.gt.get_containing_entity(entity)
                    if container and container.state.openable() and not container.state.is_open:
                        response = yield Open(container)
                    take_action = Take(entity)
                    response = yield take_action
                    if "carrying too many" in response or \
                      not gi.gt.inventory.has_entity_with_name(entityName):
                        print(f"GTAcquire: Take({entityName}) failed, try dropping something...")
                        if len(gi.gt.inventory.entities) > 0:
                            not_needed = None
                            for name in self._no_longer_needed:
                                if gi.gt.inventory.has_entity_with_name(name):
                                    not_needed = name
                                    drop_entity = gi.gt.inventory.get_entity_by_name(not_needed)
                                    break
                            if not_needed is not None:
                                self._no_longer_needed.remove(not_needed)
                            else:  # try temporarily dropping a random item from our inventory
                                drop_entity = random.choice(gi.gt.inventory.entities)
                                print("-- RANDOMLY CHOOSING to drop:", drop_entity.name)
                                #NOTE: to disabling auto-reacquire ---> # self._temporary_drop = drop_entity.name
                                # self._temporary_drop = drop_entity.name
                            self.remove_required_obj(drop_entity.name)
                            response = yield Drop(drop_entity)
                            ###gi.event_stream.push(NoLongerNeed([drop_entity.name], groundtruth=True))
                            response = yield take_action  # try again, now that we've dropped something
                            self._eagerness = 0
                            return None # let someone else do something with the new object (e.g. cut with knife)

                        else:
                            print("CAN'T DROP ANYTHING, NOTHING IN INVENTORY!!!")
                    else:
                        print(f"SUCCESSFULLY ACQUIRED '{entityName}': {response}")
                        break
                else: # can't take it, but we've found it, so consider it acquired...
                    self.remove_required_obj(entityName)

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
                # self._eagerness = 0
        else:
            self._eagerness = 0
        return None  # terminate generator iteration

