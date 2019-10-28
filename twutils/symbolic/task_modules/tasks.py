import random

from ..event import NeedToGoTo, NeedToAcquire, NeedToFind, NeedToDo
from ..task import Task
from ..action import Action
from ..game import GameInstance


class SingleActionTask(Task):
    def __init__(self, act: Action, description=None):
        if not description:
            description = f"SingleActionTask[{str(act)}]"
        else:
            description = f"SingleActionTask['{description}']"
        super().__init__(description=description)
        self.action = act

    def check_postconditions(self, gi: GameInstance) -> bool:
        return True

    def check_result(self, result: str, gi: GameInstance) -> bool:
        return True

    def _generate_actions(self, gi) -> Action:
        """ Generates a sequence of actions.
        :type gi: GameInstance
        """
        ignored = yield   # required handshake
        all_satisfied, missing = self.check_preconditions(gi)
        if all_satisfied:
            result = yield self.action
            if self.check_result(result, gi):
                self._done = True
            else:
                self._failed = True
        else:
            if missing.required_tasks:
                for task in reversed(missing.required_tasks):
                    gi.event_stream.push(NeedToDo(task, groundtruth=self.use_groundtruth))
            elif missing.required_inventory:
                gi.event_stream.push(NeedToAcquire(missing.required_inventory, groundtruth=self.use_groundtruth))
            elif missing.required_objects:
                obj = random.choice(missing.required_objects)
                gi.event_stream.push(NeedToFind(obj, groundtruth=self.use_groundtruth))
            elif missing.required_locations:
                loc = random.choice(missing.required_locations)
                gi.event_stream.push(NeedToGoTo(loc, groundtruth=self.use_groundtruth))
        return None
