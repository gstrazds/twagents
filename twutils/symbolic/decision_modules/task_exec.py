import random
from ..decision_module import DecisionModule
from ..event import NeedToDo, NeedToAcquire, NeedToFind, NeedToGoTo, NoLongerNeed
from ..game import GameInstance
from ..task import Task, Preconditions, ParallelTasks, SequentialTasks
from ..gv import dbg

def _check_preconditions(task: Task, gi: GameInstance) -> bool:
    use_groundtruth = task.use_groundtruth
    print("_check_preconditions({}) {}:".format("GT" if use_groundtruth else 'kg', task))
    kg = gi.gt if task.use_groundtruth else gi.kg
    all_satisfied = task.check_preconditions(kg)
    if not all_satisfied:
        print(f"TaskExecutor {task} has unsatisfied preconditions:\n{task.missing}")
    else:
        print("task ... preconditions SATISFIED")
    return all_satisfied

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

    def print_state(self):
        print("[[[[[ TaskExecutor:  ]]]]]")
        print("[[       .task_stack:   ]]")
        if not self.task_stack:
            print("            EMPTY")
        for t in self.task_stack:
            print(t)
        print("[[       .task_queue:   ]]")
        if not self.task_queue:
            print("            EMPTY")
        for t in self.task_queue:
            print(t)

    def _have_a_runnable_task(self, gi: GameInstance):
        print("TaskExecutor _have_a_runnable_task?...", end='')
        all_satisfied = False
        if self.task_stack:
            active_task = self.task_stack[-1]
            all_satisfied = active_task.missing.is_empty
        return all_satisfied

    def _activate_next_task(self, gi: GameInstance):
        # assert self._action_generator is None, "Shouldn't be called if a task is already active"
        if not self.task_stack:
            if self.task_queue:
                next_task = self.task_queue.pop(0)
                self.push_task(next_task)
        if self.task_stack:
            # self._action_generator = self.task_stack[-1].generate_actions(gi)
            activating_task = self.task_stack[-1]
            if not activating_task.is_active and not activating_task.is_done:
                activating_task.activate(gi)
            if not _check_preconditions(activating_task, gi):
                print(f"_activate_next_task -- {activating_task} missing preconditions:\n{activating_task.missing}")
                self.handle_missing_preconditions(activating_task.missing, gi,
                                                  use_groundtruth=activating_task.use_groundtruth)
                activating_task.deactivate(gi)
            return self.task_stack and self.task_stack[-1].is_active
        return False

    def activate(self, gi: GameInstance):
        if True or not self._active:
            print("TaskExecutor: ACTIVATING?...", end='')
            self._activate_next_task(gi)
            if self._have_a_runnable_task(gi):
                print("ACTIVATING!")
                self._active = True
                self._eagerness = 0.7  # lower than GTAcquire -- #XXXhigher than GTEnder
            else:
                print("no runnable task, canceling TaskExecutor activation")
                self._eagerness = 0
                self._active = False

    def deactivate(self, gi: GameInstance):
        if self._active:
            print("TaskExec: DEACTIVATING.")
        self._active = False
        self._eagerness = 0.0
        # self._action_generator = None

    def get_eagerness(self, gi: GameInstance):
        print("TaskExecutor.get_eagerness => ", end='')
        """ Returns a float in [0,1] indicating how eager this module is to take control. """
        if self.task_stack or self.task_queue:  # if we have some tasks
            if not self._active or not self.task_stack or not self.task_stack[-1].is_active or self._eagerness == 0:
                # if nothing currently active, move a task from pending to active
                # if self._activate_next_task(gi):
                self.activate(gi)
        else:
            self.deactivate(gi)
        print(self._eagerness)
        # if self._eagerness == 0:
        #     self.print_state()
        return self._eagerness

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

    def start_prereq_task(self, pretask, gi: GameInstance) -> bool:
        print("start_prereq_task:", pretask)
        assert pretask not in self.task_stack
        if pretask in self.task_queue:
            # required task is already queued: activate it now
            print(f"...removing prereq task {pretask} from task_queue...")
            next_task = self.task_queue.pop(self.task_queue.index(pretask))
        else:
            next_task = pretask
        self.push_task(next_task)
        return self._activate_next_task(gi)

    def handle_missing_preconditions(self, missing: Preconditions, gi: GameInstance, use_groundtruth=False):
        # assert use_groundtruth is True
        if missing.required_tasks:
            for task in reversed(missing.required_tasks):
                print(f"handle precondition: {task}")
                self.start_prereq_task(task, gi)
        # else:
        if missing.required_locations:
            loc = random.choice(missing.required_locations)
            gi.event_stream.push(NeedToGoTo(loc, groundtruth=use_groundtruth))
        # need_to_get = []
        if missing.required_inventory:
            # need_to_get.extend(missing.required_inventory)
            gi.event_stream.push(NeedToAcquire(missing.required_inventory, groundtruth=use_groundtruth))
        if missing.required_objects:
            objname = random.choice(missing.required_objects)
            # need_to_get.append(objname)
            gi.event_stream.push(NeedToFind(objnames=[objname], groundtruth=use_groundtruth))
        # if need_to_get:  #NOTE: GTAcquire processes NeedToAcquire and NeedToGet identically
        #     gi.event_stream.push(NeedToAcquire(need_to_get, groundtruth=use_groundtruth))

    def rescind_broadcasted_preconditions(self, task, gi: GameInstance):
        use_groundtruth = task.use_groundtruth
        prereqs = task.prereq
        if prereqs.required_inventory:
            gi.event_stream.push(NoLongerNeed(prereqs.required_inventory, groundtruth=use_groundtruth))
        if prereqs.required_objects:
            gi.event_stream.push(NoLongerNeed(objnames=prereqs.required_objects, groundtruth=use_groundtruth))


    def process_event(self, event, gi: GameInstance):
        """ Process an event from the event stream. """
        # print("TaskExecutor PROCESS EVENT: ", event)
        if isinstance(event, NeedToDo):
            print("TaskExecutor PROCESS EVENT: ", event)
            self.start_prereq_task(event.task, gi)
            self.activate(gi)

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

    def take_control(self, gi: GameInstance):
        observation = yield #NoAction
        print("++++TaskExecutor.take_control")
        # assert observation is None, f"TaskExec.take_control() got initial observation={observation}"
        # print(f"TaskExec.take_control() got initial observation={observation}")
        # assert self._active, \
        if not self._active:
            f"WARNING!: TaskExec.take_control() shouldn't be happening: _active={self._active}, _eagerness={self._eagerness}"
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
            if all_satisfied or active_task.is_done:
                next_action = active_task.get_next_action(observation,gi)
            else:
                self.handle_missing_preconditions(active_task.missing, gi,
                                                  use_groundtruth=active_task.use_groundtruth)
                next_action = None

            # if not next_action:
            #     failed_counter += 1
            # if failed_counter > 10:
            #     dbg(f"[TaskExec] generate_next_action FAILED {failed_counter} times! => self.deactivate()")
            #     break
                # self.deactivate(gi)
                # return None
            print(f"[TaskExecutor] (generate_next_action) active:{str(self.task_stack[-1])} -> |{next_action}|")
            if next_action:
                observation = yield next_action
                # print(f"RECEIVED observation={observation}")
            else:
            # except StopIteration:  # current task is stopping (might be done, paused missing preconditions, or failed):
                if self.task_stack[-1].is_done:
                    t = self.pop_task()
                    print(f"    (popped because {t}.is_done)")
                    self.rescind_broadcasted_preconditions(t, gi)
                    t.deactivate(gi)
                    self.completed_tasks.append(t)
                    self._activate_next_task(gi)
                else:
                    yield None
                    failed_counter += 1
                    if failed_counter > 10:
                        print(f"[TaskExec] generate_next_action FAILED {failed_counter} times! => self.deactivate()")
                        break
                    # self.deactivate(gi)
                    # return None
        self.deactivate(gi)
        return None

