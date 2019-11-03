import random
from ..decision_module import DecisionModule
from ..action import NoAction
from ..event import NeedToDo
from ..event import GroundTruthComplete, NeedToAcquire, NoLongerNeed
from ..event import NeedSequentialSteps, NeedToGoTo, NeedToFind  #, AlreadyDone
from ..game import GameInstance
from ..task import Task, Preconditions, ParallelTasks, SequentialTasks
from ..gv import dbg

def _check_preconditions(task: Task, gi: GameInstance) -> bool:
    use_groundtruth = task.use_groundtruth
    kg = gi.gt if task.use_groundtruth else gi.kg
    return task.check_preconditions(kg)


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
        # self._action_generator = None

    def get_eagerness(self, gi: GameInstance):
        """ Returns a float in [0,1] indicating how eager this module is to take control. """
        if self.task_stack or self.task_queue:  # if we have some tasks that need to be done
            if not self.task_stack:  # if nothing currently active, move a task from pending to active
                if self._activate_next_task(gi):
                    self.activate()
        else:
            self.deactivate()
        return self._eagerness

    def activate(self):
        if not self._active:
            print("TaskExec: ACTIVATING.")
        self._active = True
        self._eagerness = 0.9

    def deactivate(self):
        if self._active:
            print("TaskExec: DEACTIVATING.")
        self._active = False
        self._eagerness = 0.0
        # self._action_generator = None

    def queue_task(self, task: Task):
        print(f"TaskExec.queue_task({task})")
        self.task_queue.append(task)

    def cancel_queued_task(self, task: Task):
        print(f"TaskExec.cancel_queued_task({task})")
        self.task_queue.discard(task)

    def push_task(self, task: Task):
        assert task not in self.task_queue
        assert task not in self.task_stack
        print(f"TaskExec.push_task({task})")
        self.task_stack.append(task)

    def pop_task(self, task: Task = None):
        popped = self.task_stack.pop()
        print(f"TaskExec.pop_task({task}) => {popped}")
        if task:
            assert task == popped
            # task.deactivate()
        # self._action_generator = None
        return popped

    def start_prereq_task(self, pretask, gi: GameInstance):
        print("start_prereq_task:", pretask)
        assert pretask not in self.task_stack
        if pretask in self.task_queue:
            # required task is already queued: activate it now
            print(f"...removing prereq task {pretask} from task_queue...")
            next_task = self.task_queue.pop(self.task_queue.index(pretask))
        else:
            next_task = pretask
        self.push_task(next_task)
        self._activate_next_task(gi)

    def handle_missing_preconditions(self, missing: Preconditions, gi: GameInstance, use_groundtruth=False):
        if missing.required_tasks:
            for task in reversed(missing.required_tasks):
                print(f"handle precondition: {task}")
                self.start_prereq_task(task, gi)
        elif missing.required_locations:
             loc = random.choice(missing.required_locations)
             gi.event_stream.push(NeedToGoTo(loc, groundtruth=use_groundtruth))
        # elif missing.required_inventory:
        #     gi.event_stream.push(NeedToAcquire(missing.required_inventory, groundtruth=self.use_groundtruth))
        # elif missing.required_objects:
        #     obj = random.choice(missing.required_objects)
        #     gi.event_stream.push(NeedToFind(obj, groundtruth=self.use_groundtruth))

    def process_event(self, event, gi: GameInstance):
        """ Process an event from the event stream. """
        print("PROCESS EVENT: ", event)
        if isinstance(event, NeedToDo):
            self.start_prereq_task(event.task, gi)

        # if isinstance(event, GroundTruthComplete) and event.is_groundtruth:
        #     print("GT complete", event)
        #     self.get_eagerness(gi)
        # if isinstance(event, NeedToAcquire) and event.is_groundtruth:
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
        # elif isinstance(event, AlreadyDone) and event.is_groundtruth:
        #     print("GT AlreadyDone:", event.instr_step)
        #     self.remove_step(event.instr_step)
        pass

    def _activate_next_task(self, gi: GameInstance):
        # assert self._action_generator is None, "Shouldn't be called if a task is already active"
        if not self.task_stack:
            if self.task_queue:
                next_task = self.task_queue.pop(0)
                self.push_task(next_task)
        if self.task_stack:
            # self._action_generator = self.task_stack[-1].generate_actions(gi)
            activating_task = self.task_stack[-1]
            activating_task.activate(gi)
            if not _check_preconditions(activating_task, gi):
                print(f"_activate_next_task -- {activating_task} missing preconditions:\n{activating_task.missing}")
                self.handle_missing_preconditions(activating_task.missing, gi,
                                                  use_groundtruth=activating_task.use_groundtruth)
            return True
        return False

    def take_control(self, gi: GameInstance):
        observation = yield #NoAction
        print("++++TaskExecutor.take_control")
        print("RECEIVED:", observation)
        # assert observation is None, f"TaskExec.take_control() got initial observation={observation}"
        print(f"TaskExec.take_control() got initial observation={observation}")
        assert self._active, \
            f"TaskExec.take_control() shouldn't be happening: _active={self._active}, _eagerness={self._eagerness}"
        if not self._active:
            self._eagerness = 0.0
            return None  #ends the generator
        self._activate_next_task(gi)
        # assert self.task_stack, "TaskExec shouldn't be active if no Task is active"
        failed_counter = 0
        while self.task_stack and self.task_stack[-1].is_active:
            # try:
            active_task = self.task_stack[-1]
            all_satisfied = _check_preconditions(active_task, gi)
            if all_satisfied:
                next_action = active_task.get_next_action(observation,gi)
            else:
                print(f"TaskExecutor {active_task} has unsatisfied preconditions:\n{active_task.missing}")

                self.handle_missing_preconditions(active_task.missing, gi,
                                                  use_groundtruth=active_task.use_groundtruth)
                next_action = None

            # if not next_action:
            #     failed_counter += 1
            # if failed_counter > 10:
            #     dbg(f"[TaskExec] generate_next_action FAILED {failed_counter} times! => self.deactivate()")
            #     break
                # self.deactivate()
                # return None
            print(f"[TaskExecutor] (generate_next_action) active:{str(self.task_stack[-1])} -> |{next_action}|")
            if next_action:
                observation = yield next_action
                print(f"RECEIVED observation={observation}")
            else:
            # except StopIteration:  # current task is stopping (might be done, paused missing preconditions, or failed):
                if self.task_stack[-1].is_done:
                    t = self.pop_task()
                    print(f"    (popped because {t}.is_done)")
                    t.deactivate(gi)
                    self.completed_tasks.append(t)
                    self._activate_next_task(gi)
                else:
                    yield None
                    failed_counter += 1
                    if failed_counter > 10:
                        print(f"[TaskExec] generate_next_action FAILED {failed_counter} times! => self.deactivate()")
                        break
                    # self.deactivate()
                    # return None
        self.deactivate()
        return None

