import random
from ..decision_module import DecisionModule
from ..action import Take, Drop, Open, Portable
from ..event import NeedToAcquire, NeedToFind, NeedToGoTo, NoLongerNeed
from ..game import GameInstance
from .. import gv


class GTAcquire(DecisionModule):
    """ The Ground Truth Acquire module locates and, for takeable objects, takes specific objects (into inventory) """
    def __init__(self, active=False, use_groundtruth=True):
        super().__init__()
        self.required_objs = set()
        self.found_objs = set()
        self._active = active
        self._temporary_drop = None  # remember an object that we need to re-acquire
        self._no_longer_needed = set()
        self.use_groundtruth = use_groundtruth

    @property
    def maybe_GT(self):
        return "GT " if self.use_groundtruth else ""

    def _knowledge_graph(self, gi):
        if self.use_groundtruth:
            return gi.gt
        return gi.kg


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
        if event.is_groundtruth == self.use_groundtruth:
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
        self.record(success)

    def take_control(self, gi: GameInstance):
        ignored_obs = yield
        if self._temporary_drop:
            gi.event_stream.push(NeedToAcquire([self._temporary_drop], groundtruth=self.use_groundtruth))
            self._temporary_drop = None
            yield None  # reevaluate which should be the active module

        # TODO: FIX THIS HACK: open every container
        current_loc = self._knowledge_graph(gi).player_location
        ## if self.open_all_containers:
        for entity in list(current_loc.entities):
            # print(f"GTAcquire -- {current_loc} {entity} is_container:{entity.is_container}")
            if entity.is_container and entity.state.openable:
                print(f"GTAcquire -- {current_loc} {entity}.is_open:{entity.state.is_open}")
            if entity.is_container and entity.state.openable and not entity.state.is_open:
                response = yield Open(entity)
                entity.open()

        still_needed = list(self.missing_objs(self._knowledge_graph(gi)))
        print("GT_Acquire required_objs:", self.required_objs)
        print("GT_Acquire still_needed:", still_needed)
        here = self._knowledge_graph(gi).player_location
        for entityName in still_needed:
            if self._knowledge_graph(gi).inventory.has_entity_with_name(entityName):
                assert False, f"Still needed object {entityName} *should not be* in Inventory"
            elif self._knowledge_graph(gi).location_of_entity_with_name(entityName) is here:
                entity = self._knowledge_graph(gi).get_entity(entityName)
                if Portable in entity.attributes:
                    container = self._knowledge_graph(gi).get_holding_entity(entity)
                    if container and container.state.openable and not container.state.is_open:
                        response = yield Open(container)
                    take_action = Take(entity)
                    response = yield take_action
                    if "carrying too many" in response or \
                      not self._knowledge_graph(gi).inventory.has_entity_with_name(entityName):
                        print(f"GTAcquire: Take({entityName}) failed, try dropping something...")
                        if len(self._knowledge_graph(gi).inventory.entities) > 0:
                            not_needed = None
                            for name in self._no_longer_needed:
                                if self._knowledge_graph(gi).inventory.has_entity_with_name(name):
                                    not_needed = name
                                    drop_entity = self._knowledge_graph(gi).inventory.get_entity_by_name(not_needed)
                                    break
                            if not_needed is not None:
                                self._no_longer_needed.remove(not_needed)
                            else:  # try temporarily dropping a random item from our inventory
                                drop_entity = random.choice(self._knowledge_graph(gi).inventory.entities)
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

        still_needed = list(self.missing_objs(self._knowledge_graph(gi)))
        if still_needed:
            whereabouts_unknown = []
            for obj_to_find in still_needed:
            # obj_to_find = still_needed[0]   # TODO: make it smarter - choose closest instead of first
                loc = self._knowledge_graph(gi).location_of_entity_with_name(obj_to_find)
                if loc:
                    print("GT_Acquire NeedToGoTo:", loc)
                    gi.event_stream.push(NeedToGoTo(loc.name, groundtruth=self.use_groundtruth))
                    break
                else:
                    print(f"Don't know {self.maybe_GT}location of {obj_to_find}")
                    whereabouts_unknown.append(obj_to_find)
                    # assert False, f"Don't know {self.maybe_GT}location of {obj_to_find}"
            if self.use_groundtruth:
                for obj_to_find in whereabouts_unknown:
                    self.remove_required_obj(obj_to_find)
                    gi.event_stream.push(NoLongerNeed([obj_to_find], groundtruth=self.use_groundtruth))
            elif len(still_needed) == len(whereabouts_unknown):  # we don't know where any of the sought objects are
                gi.event_stream.push(NeedToGoTo(f'TryToFind({whereabouts_unknown[0]})', groundtruth=self.use_groundtruth))
                # will invoke GTNavigator.set_goal_by_name() to navigate to closest UnknownLocation
            # self._eagerness = 0
        else:
            self._eagerness = 0
        return None  # terminate generator iteration

