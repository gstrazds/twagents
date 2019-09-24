from ..decision_module import DecisionModule
from ..event import GroundTruthComplete, NeedToAcquire, NoLongerNeed
from ..event import NeedSequentialSteps, AlreadyAcquired, NeedToGoTo, NeedToFind
from ..game import GameInstance
from ..task import Task, ParallelTasks, SequentialTasks
from ..gv import dbg


class TaskExecutor(DecisionModule):
    """
    The TaskExecutor module sequences and executes Tasks.
    """
    def __init__(self, active=False):
        super().__init__()
        self._active = active
        self._eagerness = 0.0
        self.found_objs = set()
        self.task_stack = []  # stack of currently active tasks (of which the top most is currently in control)
        self.task_queue = []  # tasks waiting to begin
        self.completed_tasks = []
        self._action_generator = None

    def get_eagerness(self, gi: GameInstance):
        """ Returns a float in [0,1] indicating how eager this module is to take control. """
        if self.task_stack or self.task_queue:  # if we have some tasks that need to be done
            if not self.task_stack:  # if nothing currently active, move a task from pending to active
                self._activate_next_task(gi)
            self.activate()
        else:
            self.deactivate()
        return self._eagerness

    def activate(self):
        if not self._active:
            print("TaskExec: ACTIVATING.")
        self._active = True
        self._eagerness = 0.5

    def deactivate(self):
        if self._active:
            print("TaskExec: DEACTIVATING.")
        self._active = False
        self._eagerness = 0.0
        self._action_generator = None

    def queue_task(self, task: Task):
        print(f"TaskExec.queue_task({task})")
        self.task_queue.append(task)

    def cancel_queued_task(self, task: Task):
        print(f"TaskExec.cancel_queued_task({task})")
        self.task_stack.discard(task)

    def push_task(self, task: Task):
        assert task not in self.task_queue
        assert task not in self.task_stack
        print(f"TaskExec.push_task({task})")
        self.task_stack.append(task)

    def pop_task(self, task: Task = None):
        popped = self.task_queue.pop()
        if task:
            assert task == popped
        self._action_generator = None
        return popped

    def process_event(self, event, gi: GameInstance):
        """ Process an event from the event stream. """
        # if isinstance(event, GroundTruthComplete) and event.is_groundtruth:
        #     print("GT complete", event)
        #     self.get_eagerness(gi)
        # elif isinstance(event, NeedToAcquire) and event.is_groundtruth:
        #     print("GT Required Objects:", event.objnames)
        #     for itemname in event.objnames:
        #         # gi.gt.entities_with_name()
        #         self.add_required_obj(itemname)
        # elif isinstance(event, NeedToFind) and event.is_groundtruth:
        #     print("GT Need to Find Objects:", event.objnames)
        #     for itemname in event.objnames:
        #         # gi.gt.entities_with_name()
        #         self.add_required_obj(itemname)
        # elif isinstance(event, NoLongerNeed) and event.is_groundtruth:
        #     print("GT Not Needed Objects:", event.objnames)
        #     for itemname in event.objnames:
        #         self.remove_required_obj(itemname)
        # elif isinstance(event, NeedSequentialSteps) and event.is_groundtruth:
        #     print("GT Need To Do:", event.steps)
        #     for acttext in event.steps:
        #         self.add_step(acttext)
        # elif isinstance(event, AlreadyAcquired) and event.is_groundtruth:
        #     print("GT AlreadyAcquired:", event.instr_step)
        #     self.remove_step(event.instr_step)
        pass

    def _activate_next_task(self, gi: GameInstance):
        assert self._action_generator is None, "Shouldn't be called if a task is already active"
        if not self.task_stack:
            if self.task_queue:
                next_task = self.task_queue.pop(0)
                self.push_task(next_task)
        if self.task_stack:
            self._action_generator = self.task_stack[0].generate_actions(gi)

    def take_control(self, gi: GameInstance):
        observation = yield   # a newly activated decision module always gets a .send(None) first. Ignore it.
        assert observation is None, f"TaskExec.take_control() got initial observation={observation}"
        assert self._active, \
            f"TaskExec.take_control() shouldn't be happening: _active={self._active}, _eagerness={self._eagerness}"
        if not self._active:
            self._eagerness = 0.0
            return None  #ends the generator
        assert self.task_stack, "TaskExec shouldn't be active if no Task is active"
        self._activate_next_task(gi)
        failed_counter = 0
        while self._action_generator:
            try:
                next_action = self.action_generator.send(observation)
                if not next_action:
                    failed_counter += 1
                if failed_counter > 10:
                    dbg(f"[TaskExec] generate_next_action FAILED {failed_counter} times! => self.deactivate()")
                    break
                    # self.deactivate()
                    # return None
                print(f"[NAIL] (generate_next_action): ({str(self.task_stack[0])} -> |{next_action}|")
                if next_action:
                    observation = yield next_action
            except StopIteration:  # current task is stopping (might be .done, paused missing preconditions, or failed)
                if self.task_stack[0].done:
                    self.completed_tasks.append(self.pop_task())
                    self._activate_next_task(gi)
        self.deactivate()
        return None

