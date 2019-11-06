from typing import List
from ..event import NeedToGoTo, NeedToAcquire, NeedToFind, NeedToDo
from ..task import Task
from ..action import Action
from ..game import GameInstance


class SequentialActionsTask(Task):
    def __init__(self, actions: List[Action], description=None, use_groundtruth=False):
        assert actions, "Required arg: must specify at least one Action"
        self.actions = actions
        self._current_idx = 0
        if not description:
            actions_desc = ','.join([str(act) for act in actions])
            description = f"SequentialActionsTask[{actions_desc}]"
        else:
            description = description
        super().__init__(description=description, use_groundtruth=use_groundtruth)

    def reset(self):
        self._current_idx = 0
        super().reset()

    def check_postconditions(self, gi: GameInstance) -> bool:
        return True

    def check_result(self, result: str, gi: GameInstance) -> bool:
        return True

    def _generate_actions(self, gi) -> Action:
        """ Generates a sequence of actions.
        :type gi: GameInstance
        """
        ignored = yield   # required handshake
        while self._current_idx < len(self.actions) and not self._failed:
            result = yield self.actions[self._current_idx]
            self._current_idx += 1
            if self.check_result(result, gi):
                if self._current_idx >= len(self.actions):
                    self._done = True
            else:
                self._failed = True
        return None


class SingleActionTask(SequentialActionsTask):
    def __init__(self, act: Action, description=None, use_groundtruth=False):
        actions = [act]
        if not description:
            description = f"SingleActionTask[{str(act)}]"
        else:
            description = f"SingleActionTask['{description}']"
        super().__init__(actions=actions, description=description, use_groundtruth=use_groundtruth)

    @property
    def action(self):
        return self.actions[0] if self.actions else None

    def _generate_actions(self, gi) -> Action:
        """ Generates a sequence of actions.
        :type gi: GameInstance
        """
        ignored = yield   # required handshake
        while self._current_idx < len(self.actions) and not self._failed:
            self._done = True  #FIXME: HACK to shortcircuit preconditions check for next iteration
            result = yield self.actions[self._current_idx]
            self._current_idx += 1
            if not self.check_result(result, gi):
                self._failed = True
        return None
