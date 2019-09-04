from typing import List

class Preconditions:
    def __init__(self):
        self.required_inventory = []  # items that must be in Inventory for this task to succeed
        self.required_objects = []    # non-takeable objects: need to be near one of these
        self.required_locations = []   # need to be at one of these locations
        self.required_tasks = []   # list of tasks that need to be done before this one should be attempted


class Task:
    """ Base class for Tasks. """
    def __init__(self, description=''):
        self._done = False
        self.description = description
        self.prereq = Preconditions()

    @property
    def done(self) -> bool:
        return self._done

    def reset(self):
        self._done = False

    def check_preconditions(self, kg) -> (bool, Preconditions):
        missing = Preconditions()
        if self.prereq.required_inventory:
            for name in self.prereq.required_inventory:
                if not kg or not kg.inventory.has_entity_with_name(name):
                    missing.required_inventory.append(name)
        if self.prereq.required_objects:
            for name in self.prereq.required_inventory:
                if kg and kg.player_location.has_entity_with_name(name):
                    missing.required_objects.clear()
                    break  # we only need one: declare that none are missing
                else:
                    missing.required_objects.append(name)
        if self.prereq.required_locations and \
                (not kg or kg.player_location.name not in self.prereq.required_locations):
            missing.required_locations += self.prereq.required_locations
        # all(map(lambda t: t.done, self.prereq_tasks))
        for t in self.prereq.required_tasks:
            if not t.done:
                missing.required_tasks.append(t)
        all_satisfied = \
            not missing.required_inventory and \
            not missing.required_objects and \
            not missing.required_locations and \
            not missing.required_tasks
        return all_satisfied, missing


    def __str__(self):
        return self.description

    def __repr__(self):
        return str(self)


class CompositeTask(Task):
    def __init__(self, tasks: List[Task], description=None):
        if not description:
            description = "{classname}{tasklist}".format(
                classname=type(self).__name__, tasklist=str([t for t in tasks]))
        super().__init__(description=description)
        self.tasks = tasks

    @property
    def done(self) -> bool:
        if not self._done:  #one-way caching: once it's done, it stays done (until reset() is called)
            self._done = self.check_done()
        return self._done

    def check_done(self) -> bool:
        return all(map(lambda t: t.done, self.tasks))

    def reset_all(self):
        for t in self.tasks:
            t.reset()
        self._done = False

    def reset(self):
        print(f"WARNING: reset() instead of reset_all() on <{str(self)}> ?")
        super().reset()


class SequentialTasks(CompositeTask):
    def __init__(self, tasks: List[Task], description=None):
        super().__init__(tasks, description=description)

    @property
    def done(self) -> bool:
        return super().done


class ParallelTasks(CompositeTask):
    def __init__(self, tasks: List[Task], description=None):
        super().__init__(tasks, description=description)

    @property
    def done(self) -> bool:
        return super().done
