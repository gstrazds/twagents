from typing import List
from .navigation_task import GoToTask
from ..action import Take, Drop, Open, Portable
# from ..game import GameInstance
from ..task import Task, SequentialTasks
from ..action import Action
from ..gv import rng, dbg


class TakeItemTask(Task):
    def __init__(self, item_name, description='', use_groundtruth=True):
        def _item_is_in_inventory(kgraph): #closure (captures self._item_name) for postcondition check
            retval = kgraph.inventory.has_entity_with_name(self._item_name)
            print(f"POSTCONDITION item_is_in_inventory({self._item_name}) => {retval}")
            return retval

        assert item_name, "REQUIRED ARGUMENT: Must specify the name of an item to acquire"
        if not description:
            description = f"TakeTask[{item_name}]"

        super().__init__(description=description, use_groundtruth=use_groundtruth)
        self._item_name = item_name
        self.add_postcondition(_item_is_in_inventory)
        self.prereq.add_required_object(item_name)  # need to be in the vicinity of this object

    def _generate_actions(self, kg) -> Action:
        """ Generates a sequence of actions.
        :type kg: KnowledgeGraph
        """
        #NOTE: at this point, preconditions have been met, and postconditions are not currently all satisfied
        ignored = yield   # required handshake
        # here = self._knowledge_graph(gi).player_location
        here = kg.player_location
        entityName = self._item_name
        assert kg.location_of_entity_with_name(entityName) is here
        entity = kg.get_entity(entityName)
        # assert Portable in entity.attributes
        container = kg.get_holding_entity(entity)
        if container and container.state.openable and not container.state.is_open:
            response = yield Open(container)
        take_action = Take(entity)
        response = yield take_action
        if "carrying too many" in response \
           or "need to drop" in response \
           or not kg.inventory.has_entity_with_name(entityName):
            print(f"TakeItemTask: Take({entityName}) failed, try dropping something...")
            if len(kg.inventory.entities) > 0:
                drop_entity = rng.choice(kg.inventory.entities)
                print("-- RANDOMLY CHOOSING to drop:", drop_entity.name)
                response2 = yield Drop(drop_entity)
                response = yield take_action  # try again, now that we've dropped something
                print(f"{self} Responses for DROP: [{response2}]; TAKE2: [{response}]")
            else:
                print("CAN'T DROP ANYTHING, NOTHING IN INVENTORY!!!")
                self._failed = True
        self._failed = not self.check_postconditions(kg, deactivate_ifdone=True)
        if not self._failed:
            print(f"SUCCESSFULLY ACQUIRED '{entityName}': {response}")
        return None  # terminate generator iteration


class AcquireTask(SequentialTasks):
    """ The Ground Truth Acquire module locates takeable objects, takes specific objects (into inventory) """
    def __init__(self, objname=None, description=None, take=True, use_groundtruth=False):
        assert objname, "Missing required argument: must specify the name of an object to acquire"
        self.take_when_found = take

        task_list: List[Task] = []
        if not description:
            description = f"AcquireTask[{objname}]"
        super().__init__(tasks=task_list, description=None, use_groundtruth=False)

    def activate(self, kg):
        print("PathTask.activate!!!!")
        if self.set_goal(kg) and self.path:  # might auto self.deactivate(kg)
            return super().activate(kg)
        else:
            self.deactivate(kg)
            return None

    @property
    def maybe_GT(self):
        return "GT " if self.use_groundtruth else ""

