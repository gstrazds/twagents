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
        if kg is None:
            print("WARNING: Preconditions.check_current_state(kg==None)")
        missing = Preconditions()
        if self.required_inventory:
            for name in self.required_inventory:
                if not kg or not kg.inventory.has_entity_with_name(name):
                    missing.required_inventory.append(name)
        if self.required_objects:
            for name in self.required_objects:
                if kg and kg.player_location.has_entity_with_name(name):
                    missing.required_objects.clear()
                    break  # we only need one: declare that none are missing
                else:
                    missing.required_objects.append(name)
        if self.required_locations and \
                (not kg or kg.player_location.name not in self.required_locations):
            missing.required_locations += self.required_locations
        # all(map(lambda t: t.is_done, self.prereq_tasks))
        for t in self.required_tasks:
            if not t.is_done:
                missing.required_tasks.append(t)
        return missing


class Task:
    """ Base class for Tasks. """
    def __init__(self, description='', use_groundtruth=False):
        self._done = False
        self._failed = False
        self.use_groundtruth = use_groundtruth
        self.description = description
        self.prereq = Preconditions()
        self.missing = Preconditions()
        self._action_generator = None  # generator: current state of self._generate_actions()

    @property
    def is_done(self) -> bool:
        return self._done

    @property
    def is_active(self) -> bool:
        return self._action_generator is not None

    @property
    def has_failed(self) -> bool:
        return self._failed

    def reset(self):
        self._done = False
        self._failed = False
        self._action_generator = None

    def check_preconditions(self, kg) -> (bool, Preconditions):
        self.missing = self.prereq.check_current_state(kg)
        return self.missing.is_empty

    def _generate_actions(self, gi) -> Action:
        """ Generates a sequence of actions.
        :type gi: GameInstance
        """
        ignored = yield
        return None

    def activate(self, gi):
        if not self.is_active:
            print(f"{self} ACTIVATING")
            self._action_generator = self._generate_actions(gi)  #proxy waiting for.send(obs) at initial "ignored = yield"
            self._action_generator.send(None)
        return self._action_generator

    def deactivate(self, gi):
        if self.is_active:
            print(f"{self} DEACTIVATING")
            self._action_generator = None

    def get_next_action(self, observation, gi) -> Action:
        # act = None
        gen = self._action_generator
        if gen:
            try:
                act = gen.send(observation)
            except StopIteration:
                act = None
            if not act:
                self.deactivate(gi)
        else:
            errmsg = f"get_next_action() called for inactive task {self}"
            print(f"ERROR: "+errmsg)
            assert False, errmsg
        return act

    def __str__(self):
        return "{} active:{} done:{} failed:{}".format(
            self.description, self.is_active, self.is_done, self.has_failed)

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
    def is_done(self) -> bool:
        if not self._done:  #one-way caching: once it's done, it stays done (until reset() is called)
            self._done = self._check_done()
        return self._done

    def _check_done(self) -> bool:
        return all(map(lambda t: t.is_done, self.tasks))

    def reset_all(self):
        for t in self.tasks:
            t.reset()
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

    def get_current_task(self, gi):
        if self.tasks and 0 <= self._current_idx < len(self.tasks):
            return self.tasks[self._current_idx]
        # print("SequentialTasks.get_current_task(idx={}) => None (tasks:{})".format(self._current_idx, self.tasks))
        return None

    def activate(self, gi):
        t = self.get_current_task(gi)
        # self._action_generator = self._generate_actions(gi) #proxy waiting for.send(obs) at initial "ignored = yield"
        self._action_generator = t.activate(gi) if t else None
        return self._action_generator

    def deactivate(self, gi):
        t = self.get_current_task(gi)
        if t:
            t.deactivate(gi)
        super().deactivate(gi)

    @property
    def is_done(self) -> bool:
        if self._current_idx > 0 and len(self.tasks) > 0 and not self._done:
            for idx in range(self._current_idx):
                assert idx < len(self.tasks)
                assert self.tasks[idx].is_done
        return super().is_done

    def get_next_action(self, observation, gi) -> Action:
        """ Generates a sequence of actions.
        SequentialTask simply invokes the corresponding method on the currently active subtask."""
        if self._done: #shortcut, maybe not needed?
            return None
        t = self.get_current_task(gi)
        act = self.tasks[self._current_idx].get_next_action(observation, gi)
        if act:
            return act
        else:
            if self.tasks[self._current_idx].is_done:
                self._current_idx += 1  # move on to the next task, if there is one
                if self._current_idx < len(self.tasks):
                    self.activate(gi)  # reactivate with new current task
                    return self.get_next_action(observation, gi)  # RECURSE to next Task
                else:
                    self._done = True
                    self.deactivate(gi)
            else:  # current task stopped but is incomplete (failed, at least for now)
                self.deactivate(gi)  #self.suspend(gi)
        return None


class ParallelTasks(CompositeTask):
    def __init__(self, tasks: List[Task], description=None, use_groundtruth=False):
        super().__init__(tasks, description=description, use_groundtruth=use_groundtruth)

    @property
    def is_done(self) -> bool:
        return super().is_done
