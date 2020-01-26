# from abc import ABC, abstractmethod
from typing import List
from .action import Action


class Preconditions:
    def __init__(self):
        self.required_inventory = []  # *all* of these items must be in Inventory for this task to succeed
        self.required_objects = []    # non-takeable objects: need to be near *one* of these
        self.required_locations = []   # need to be at one of these locations
        self.required_tasks = []   # list of tasks that need to be done before this one should be attempted

    def __str__(self):
        desc_str = ""
        if self.required_inventory:
            desc_str += f"inventory: {self.required_inventory}\n"
        if self.required_objects:
            desc_str += f"objects: {self.required_objects}\n"
        if self.required_locations:
            desc_str += f"locations: {self.required_locations}\n"
        if self.required_tasks:
            desc_str += f"tasks: {self.required_tasks}\n"
        return desc_str[:-1]

    def add_required_item(self, item_name: str):
        if item_name not in self.required_inventory:
            self.required_inventory.append(item_name)
            return True
        else:
            return False

    def add_required_location(self, location_name: str):
        if location_name not in self.required_locations:
            self.required_locations.append(location_name)
            return True
        else:
            return False

    def add_required_object(self, object_name: str):
        if object_name not in self.required_objects:
            self.required_objects.append(object_name)
            return True
        else:
            return False

    def add_required_task(self, task):
        if task not in self.required_tasks:
            self.required_tasks.append(task)
            return True
        else:
            return False

    @property
    def is_empty(self) -> bool:
        all_empty = \
            not self.required_inventory and \
            not self.required_objects and \
            not self.required_locations and \
            not self.required_tasks
        return all_empty

    def check_current_state(self, kg):
        if kg:
            here = kg.player_location
        else:
            print("WARNING: Preconditions.check_current_state(kg==None)")
            here = None
        missing = Preconditions()
        if self.required_objects:
            for name in list(self.required_objects): # NOTE: we copy the list to ensure safe removal while iterating
                if kg and kg.location_of_entity_with_name(name) == here:
                    if kg.is_object_portable(name):
                        if name not in self.required_inventory:
                            print(f"POSSIBLY NEED TO transfer '{name}' => required_inventory")
                            # self.required_inventory.append(name)
                            # self.required_objects.remove(name)
                        # print(f"Reclassifying '{name}' as portable")
                        # self.required_objects.remove(name)
                    # else:
                    missing.required_objects.clear()
                    break  # we only need to find one of several possibilities: declare that none are missing
                elif kg and kg.inventory.has_entity_with_name(name):  # portable object was misclassified as non-portable
                    print(f"portable object {name} misclassified as non-portable?")
                    # missing.required_objects.clear(); break   # this counts as having found the correct object
                else:
                    missing.required_objects.append(name)
        if self.required_inventory:
            for name in self.required_inventory:
                if not kg or not kg.inventory.has_entity_with_name(name):
                    missing.required_inventory.append(name)
        if self.required_locations and \
                (not kg or kg.player_location.name not in self.required_locations):
            missing.required_locations += self.required_locations
        # all(map(lambda t: t.is_done, self.prereq_tasks))
        for t in self.required_tasks:
            if not t.check_postconditions(kg) or not t.is_done or t.has_failed:
                missing.required_tasks.append(t)
        return missing


class Task:
    """ Base class for Tasks. """
    def __init__(self, description='', use_groundtruth=False):
        self._done = False
        self._failed = False
        self.use_groundtruth = use_groundtruth
        if not description:
            description = "{classname}".format(classname=type(self).__name__)
        self.description = description
        self.prereq = Preconditions()
        self.missing = Preconditions()
        self._postcondition_checks = []  # closures, invoked with one arg = KnowledgeGraph
        self._action_generator = None  # generator: current state of self._generate_actions()
        self._task_exec = None   # when a task is activated, it gets a pointer to the TaskExecutor
        self._children = []   # keep track of subtasks that need to be revoked if this task fails or is revoked

    def _knowledge_graph(self, gi):
        if self.use_groundtruth:
            return gi.gt
        return gi.kg

    @property
    def is_done(self) -> bool:
        if not self._done:  #one-way caching: once it's done, it stays done (until reset() is called)
            self._done = self._check_done()
        return self._done

    @property
    def is_active(self) -> bool:
        return self._action_generator is not None

    @property
    def has_failed(self) -> bool:
        return self._failed

    @property
    def has_postcondition_checks(self) -> bool:
        return self._postcondition_checks and len(self._postcondition_checks) > 0

    @property
    def subtasks(self):
        subtask_list = []
        subtask_list += self.prereq.required_tasks
        subtask_list += self._children
        return subtask_list

    def _check_done(self) -> bool:
        return self._done

    def reset(self):
        self._done = False
        self._failed = False
        self._action_generator = None

    def reset_all(self):
        for t in self._children:
            t.reset()
        self.reset()

    def check_preconditions(self, kg) -> bool:
        # if gi:
        #     kg = self._knowledge_graph(gi)
        # else:
        if not kg:
            print("WARNING: Task({self}).check_preconditions called with gi=None")
            # kg = None
        self.missing = self.prereq.check_current_state(kg)
        return self.missing.is_empty

    def check_postconditions(self, kg, deactivate_ifdone=True) -> bool:  # True if postconditions satisfied
        all_satisfied = True
        if self._postcondition_checks:
            # kg = self._knowledge_graph(gi)
            # print(f"CHECKING POSTCONDITIONS: {self} /{self._postcondition_checks}/")
            for func in self._postcondition_checks:
                if not func(kg):
                    # print("POSTCONDITION CHECK failed", self)
                    all_satisfied = False
                    break
            if all_satisfied:
                print(f"{self}: All postcondition checks passed!")
            if all_satisfied and deactivate_ifdone:
                print(f"{self} auto-deactivating because postconditions are satisfied!")
                self.deactivate(kg)
                self._done = True
        return all_satisfied

    def add_postcondition(self, check_func):
        if check_func not in self._postcondition_checks:
            self._postcondition_checks.append(check_func)
            print(f"ADDED POSTCONDITION CHECK: {self} /{self._postcondition_checks}/")
        else:
            assert False, f"WTF? {self._postcondition_checks} {check_func}"

    def _generate_actions(self, kg) -> Action:
        """ Generates a sequence of actions.
        :type kg: KnowledgeGraph
        """
        ignored = yield
        return None

    def activate(self, kg, exec):
        if self._task_exec:
            assert exec is self._task_exec
        self._task_exec = exec

        if not self.is_active:
            print(f"{self} ACTIVATING")
            self._action_generator = self._generate_actions(kg)  #proxy waiting for.send(obs) at initial "ignored = yield"
            self._action_generator.send(None)
        if kg:
            self.check_postconditions(kg, deactivate_ifdone=True)
        return self._action_generator

    def deactivate(self, kg):
        if self.is_active:
            print(f"{self} DEACTIVATING")
            self._action_generator = None

    def get_next_action(self, observation, kg) -> Action:
        # act = None
        gen = self._action_generator
        if gen:
            try:
                act = gen.send(observation)
            except StopIteration:
                act = None
            if not act:
                self.deactivate(kg)
                # self.deactivate(self._knowledge_graph(gi))
        else:
            errmsg = f"get_next_action() called for inactive task {self}"
            print(f"ERROR: "+errmsg)
            assert False, errmsg
        return act

    def __str__(self):
        return "{}({}{}{})".format(
            self.description,
            "ACTIVE" if self.is_active else "idle",
            " DONE" if self.is_done else '',
            " FAILED" if self.has_failed else '')

    def __repr__(self):
        return str(self)


class CompositeTask(Task):
    def __init__(self, tasks: List[Task], description=None, use_groundtruth=False):
        if not description:
            description = "{classname}{tasklist}".format(
                classname=type(self).__name__, tasklist=str([t for t in tasks]))
        super().__init__(description=description, use_groundtruth=use_groundtruth)
        self.tasks = tasks

    @property
    def subtasks(self):
        subtask_list = super().subtasks
        if self.tasks:
            subtask_list += self.tasks
        return subtask_list

    def _check_done(self) -> bool:
        if self.tasks:
            return all(map(lambda t: t.is_done, self.tasks))
        else:
            return super()._check_done()

    def reset_all(self):
        for t in self.tasks:
            t.reset()
        super().reset()
        self._done = False

    def reset(self):
        print(f"WARNING: reset() instead of reset_all() on <{str(self)}> ?")
        super().reset()


class SequentialTasks(CompositeTask):
    def __init__(self, tasks: List[Task], description=None, use_groundtruth=False):
        super().__init__(tasks, description=description, use_groundtruth=use_groundtruth)
        self._current_idx = 0

    def reset_all(self):
        self._current_idx = 0
        super().reset_all()

    def reset(self):
        self._current_idx = 0
        super().reset()

    def get_current_task(self, kg):
        if self.tasks and 0 <= self._current_idx < len(self.tasks):
            task = self.tasks[self._current_idx]
            task.check_postconditions(kg, deactivate_ifdone=True)
            return task
        # print("SequentialTasks.get_current_task(idx={}) => None (tasks:{})".format(self._current_idx, self.tasks))
        return None

    def activate(self, kg, exec):
        t = self.get_current_task(kg)  #self._knowledge_graph(gi))
        # self._action_generator = self._generate_actions(gi) #proxy waiting for.send(obs) at initial "ignored = yield"
        self._action_generator = t.activate(kg, exec) if t else None
        return self._action_generator

    def deactivate(self, kg):
        t = self.get_current_task(kg)
        if t:
            t.deactivate(kg)
        super().deactivate(kg)

    @property
    def is_done(self) -> bool:
        if self._current_idx > 0 and len(self.tasks) > 0 and not self._done:
            for idx in range(self._current_idx):
                assert idx < len(self.tasks)
                assert self.tasks[idx].is_done
        return super().is_done

    def get_next_action(self, observation, kg) -> Action:
        """ Generates a sequence of actions.
        SequentialTask simply invokes the corresponding method on the currently active subtask."""
        if self._done: #shortcut, maybe not needed?
            return None
        t = self.get_current_task(kg)
        act = None
        if t: # and not t.is_done:
            act = t.get_next_action(observation, kg)
        if act:
            return act
        else:
            if self.tasks[self._current_idx].is_done:
                self._current_idx += 1  # move on to the next task, if there is one
                if self._current_idx < len(self.tasks):
                    self.activate(kg, self._task_exec)  # reactivate with new current task
                    return self.get_next_action(observation, kg)  # RECURSE to next Task
                else:
                    self._done = True
                    self.deactivate(kg)
            else:  # current task stopped but is incomplete (failed, at least for now)
                self.deactivate(kg)  #self.suspend(gi)
        return None


class ParallelTasks(CompositeTask):
    """ Groups several subtasks in parallel: if activated, any of the subtasks might generate the next low-level action,
    depending on runnability (preconditions are satisfied). If more than one can run, one is chosen
    randomly (or potentially based on a priority value); it then continues generating actions until it
    completes, fails, or becomes not-runnable.

    The num_required argument determines when the overall task is considered completed.
    If num_required=0 (the default) then the ParallelTask is considered complete when *all* of the subtasks are complete.
    If num_required=1, then the ParallelTask is considered complete if any one of the subtasks completes successfully.
    (num_required > 1 is also possible, e.g. specifying that 2, or 3, or n of N subtasks are required to complete).
    num_required < 0: with N subtasks, N + num_nequired need to complete. (Error if not N > -num_required)
    """

    def __init__(self, tasks: List[Task], num_required=0, description=None, use_groundtruth=False):
        super().__init__(tasks, description=description, use_groundtruth=use_groundtruth)
        self.num_required = num_required

    def _generate_actions(self, kg) -> Action:
        assert False, f"{self} - Not Yet Implemented"

    @property
    def is_done(self) -> bool:
        return super().is_done
