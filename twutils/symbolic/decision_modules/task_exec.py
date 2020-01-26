import random
from ..decision_module import DecisionModule
from ..event import NeedToDo, NeedToAcquire, NeedToFind, NeedToGoTo, NoLongerNeed
from ..game import GameInstance
from ..task import Task, Preconditions, ParallelTasks, SequentialTasks
from ..task_modules.navigation_task import PathTask
from ..gv import dbg

def _get_kg_for_task(task: Task, gi: GameInstance):
    if not gi:
        return None
    return gi.gt if task.use_groundtruth else gi.kg

def _check_preconditions(task: Task, gi: GameInstance) -> bool:
    use_groundtruth = task.use_groundtruth
    print("_check_preconditions({}) {}:".format("GT" if use_groundtruth else 'kg', task))
    kg = _get_kg_for_task(task, gi)
    all_satisfied = task.check_preconditions(kg)
    if not all_satisfied:
        print(f"TaskExecutor {task} has unsatisfied preconditions:\n{task.missing}")
    else:
        if task.prereq.is_empty:
            print(f"task {task} has no prereqs")
        else:
            print(f"task {task}: all preconditions {task.prereq} SATISFIED")
    return all_satisfied

class TaskExecutor(DecisionModule):
    """
    The TaskExecutor module sequences and executes Tasks.
    """
    def __init__(self, active=False):
        super().__init__()
        self._gi = None     # keep a pointer to caller (GameInstance) while active
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

    def _current_task_is_runnable(self):
        print("TaskExecutor _have_a_runnable_task?...", end='')
        all_satisfied = False
        if self.task_stack:
            active_task = self.task_stack[-1]
            all_satisfied = active_task.missing.is_empty
        return all_satisfied

    def _activate_next_task(self):
        gi = self._gi
        # assert self._action_generator is None, "Shouldn't be called if a task is already active"
        if not self.task_stack:
            if self.task_queue:
                next_task = self.task_queue.pop(0)
                self.push_task(next_task)
        while self.task_stack:
            # self._action_generator = self.task_stack[-1].generate_actions(gi)
            activating_task = self.task_stack[-1]
            if not activating_task.is_done and not activating_task.is_active:
                assert not activating_task.is_done, f"Unexpected: {activating_task}.is_done !"
                kg = _get_kg_for_task(activating_task, gi)
                activating_task.activate(kg, self)
            if activating_task.is_done:
                self.pop_task(activating_task)
                self.rescind_broadcasted_preconditions(activating_task)
                continue   # loop to get next potentially active task
            if not _check_preconditions(activating_task, gi):
                print(f"_activate_next_task -- {activating_task} missing preconditions:\n{activating_task.missing}")
                self.handle_missing_preconditions(activating_task.missing,
                                                  use_groundtruth=activating_task.use_groundtruth)
                kg = _get_kg_for_task(activating_task, gi)
                activating_task.deactivate(kg)
            return self.task_stack and self.task_stack[-1].is_active
        return False

    def activate(self, gi: GameInstance):
        self._gi = gi
        if True or not self._active:
            print("TaskExecutor: ACTIVATING?...", end='')
            self._activate_next_task()
            print("Activation Prechecks...")
            self.remove_completed_tasks()
            if self._current_task_is_runnable():
                if self._active:
                    print("already active.")
                    self._eagerness = 0.75  # lower than GTNavigator -- higher than GTAcquire -- #XXXhigher than GTEnder
                else:
                    self.print_state()
                    print("ACTIVATING!")
                    self._active = True
                    self._eagerness = 0.75  # lower than GTNavigator -- higher than GTAcquire -- #XXXhigher than GTEnder
            else:
                print("No runnable task, canceling TaskExecutor activation")
                self.print_state()
                self._eagerness = 0
                self._active = False

    def deactivate(self, gi: GameInstance):
        if self._active:
            print("TaskExec: DEACTIVATING.")
            self.print_state()
        self._active = False
        self._eagerness = 0.0
        self._gi = None
        # self._action_generator = None

    def remove_completed_tasks(self):
        gi = self._gi
        print("TaskExec -- remove_completed_tasks...")
        something_changed = True
        while something_changed and self.task_stack:
            something_changed = False
            # task_list = list(self.task_stack)  # copy list of potentially active tasks
            for t in self.task_stack:
                kg = _get_kg_for_task(t, gi)
                if t.has_postcondition_checks and t.check_postconditions(kg, deactivate_ifdone=True):
                    self.pop_task(task=t)   # removes one or more tasks from task_stack
                    something_changed = True
                    break   # exit inner for-loop, continue outer while-loop

    def get_eagerness(self, gi: GameInstance):
        print("TaskExecutor.get_eagerness => ", end='')
        """ Returns a float in [0,1] indicating how eager this module is to take control. """
        if self.task_stack or self.task_queue:  # if we have some tasks
            if not self._active or not self.task_stack or not self.task_stack[-1].is_active or self._eagerness == 0:
                # if nothing currently active, move a task from pending to active
                self.activate(gi)
            else:
                print(f"(already active: task={self.task_stack[-1]})", end='')
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
        if task:
            popped = task
            self.task_stack.remove(task)
        else:
            popped = self.task_stack.pop()
        print(f"TaskExec.pop_task({task}) => {popped}")
        if task:
            assert task == popped
        subtasks = popped.subtasks  #popped.subtasks
        if subtasks:
            for t in subtasks:
                if t in self.task_stack:
                    self.pop_task(task=t)
                    t.reset_all()
            # task.deactivate()
        # self._action_generator = None
        return popped

    def start_prereq_task(self, pretask) -> bool:
        # gi = self._gi
        print("start_prereq_task:", pretask)
        # assert pretask not in self.task_stack
        if pretask in self.task_queue:
            # required task is already queued: activate it now
            print(f"...removing prereq task {pretask} from task_queue...")
            next_task = self.task_queue.pop(self.task_queue.index(pretask))
        else:
            next_task = pretask
        if next_task in self.task_stack:
            print(f"WARNING: {next_task} was already in the task stack: pop & re-push")
            popped = self.pop_task(task=next_task)
        if next_task.is_done or next_task.has_failed:
            next_task.reset_all()  # try again
        self.push_task(next_task)
        return self._activate_next_task()

    def handle_missing_preconditions(self, missing: Preconditions, use_groundtruth=False):
        gi = self._gi
        if not gi:
            kg = None
        else:
            kg = gi.gt if use_groundtruth else gi.kg
        if missing.required_inventory:
            # need_to_get.extend(missing.required_inventory)
            gi.event_stream.push(NeedToAcquire(missing.required_inventory, groundtruth=use_groundtruth))
        if missing.required_locations:
            locname = list(missing.required_locations)[0]
            if not use_groundtruth and kg and kg.location_of_entity_is_known(locname):
                # gi.event_stream.push(NeedToDo(pathtask, groundtruth=use_groundtruth))
                already_in_prereqs = False
                for t in missing.required_tasks:
                    if isinstance(t, PathTask): # and t.goal_name == locname:
                        already_in_prereqs = True
                        break
                if not already_in_prereqs:
                    pathtask = PathTask(locname, use_groundtruth=use_groundtruth)
                    print(kg.location_of_entity_with_name(locname), pathtask)
                    missing.required_tasks.append(pathtask)
            else:
                gi.event_stream.push(NeedToGoTo(locname, groundtruth=use_groundtruth))
        # need_to_get = []
        if missing.required_objects:
            # need_to_get.append(objname)
            objname = random.choice(missing.required_objects)
            if not use_groundtruth and kg and kg.location_of_entity_is_known(objname):
                # gi.event_stream.push(NeedToDo(pathtask, groundtruth=use_groundtruth))
                already_in_prereqs = False
                for t in missing.required_tasks:
                    if isinstance(t, PathTask):  # and t.goal_name == objname:
                        already_in_prereqs = True
                        break
                if not already_in_prereqs:
                    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
                    pathtask = PathTask(objname, use_groundtruth=use_groundtruth)
                    print(kg.location_of_entity_with_name(objname), pathtask)
                    missing.required_tasks.append(pathtask)
            else:
                gi.event_stream.push(NeedToFind(objnames=[objname], groundtruth=use_groundtruth))
        if missing.required_tasks:
            # for task in reversed(missing.required_tasks):  # GVS 19.01.2019 question to myself: why reversed?
            # answer re: why reversed? LIFO ordering gets reversed to FIFO when all are pushed onto task_stack
            #TODO: NOTE: here we should maybe activate only one at a time (essentially: sequential tasks)
            #TODO: or else, activate them all but using a ParallelTasks context (which is not yet implemented)
            for task in reversed(missing.required_tasks):
                print(f"handle precondition: {task}")
                self.start_prereq_task(task)

    def rescind_broadcasted_preconditions(self, task):
        gi = self._gi
        use_groundtruth = task.use_groundtruth
        prereqs = task.prereq
        if prereqs.required_inventory:
            print("rescind_broadcasted_preconditions INVENTORY:", task, prereqs.required_inventory)
            gi.event_stream.push(NoLongerNeed(prereqs.required_inventory, groundtruth=use_groundtruth))
        if prereqs.required_objects:
            print("rescind_broadcasted_preconditions OBJECTS:", task, prereqs.required_objects)
            gi.event_stream.push(NoLongerNeed(objnames=prereqs.required_objects, groundtruth=use_groundtruth))

    def process_event(self, event, gi: GameInstance):
        """ Process an event from the event stream. """
        # print("TaskExecutor PROCESS EVENT: ", event)
        if isinstance(event, NeedToDo):
            print("TaskExecutor PROCESS EVENT: ", event)
            self._gi = gi
            self.start_prereq_task(event.task)
            self.activate(gi)

    def take_control(self, gi: GameInstance):
        observation = yield #NoAction
        print("++++TaskExecutor.take_control")
        assert self._gi == gi
        # assert observation is None, f"TaskExec.take_control() got initial observation={observation}"
        # print(f"TaskExec.take_control() got initial observation={observation}")
        # assert self._active, \
        if not self._active:
            f"WARNING!: TaskExec.take_control() shouldn't be happening: _active={self._active}, _eagerness={self._eagerness}"
        if not self._active:
            self._eagerness = 0.0
            return None  #ends the generator
        self._activate_next_task()
        # assert self.task_stack, "TaskExec shouldn't be active if no Task is active"
        failed_counter = 0
        while self.task_stack and self.task_stack[-1].is_active:
            # try:
            active_task = self.task_stack[-1]
            prereqs_satisfied = False
            if not active_task.is_done:
                prereqs_satisfied = _check_preconditions(active_task, gi)
            else:
                self.rescind_broadcasted_preconditions(active_task)

            if prereqs_satisfied or active_task.is_done:
                kg = _get_kg_for_task(active_task, gi)
                next_action = active_task.get_next_action(observation, kg)  # ?get next action from DONE task?
            else:
                self.handle_missing_preconditions(active_task.missing,
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
                    kg = _get_kg_for_task(t, gi)
                    self.rescind_broadcasted_preconditions(t)
                    t.deactivate(kg)
                    self.completed_tasks.append(t)
                    self._activate_next_task()
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

