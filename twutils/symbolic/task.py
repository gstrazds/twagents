from typing import List


class Task:
    """ Base class for Tasks. """
    def __init__(self, description=''):
        self._done = False
        self.description = description
        self.required_inventory = []
        self.required_objects = []  # non-takeable objects that we need to be near
        self.required_location = []  # need to be at one of these locations
        self.prereq_tasks = []   # list of tasks that need to be done before this one should be attempted

    @property
    def done(self):
        return self._done

    def reset(self):
        self._done = False

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
    def done(self):
        if not self._done:  #one-way caching: once it's done, it stays done (until reset() is called)
            self._done = self.check_done()
        return self._done

    def check_done(self):
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
    def done(self):
        return super().done


class ParallelTasks(CompositeTask):
    def __init__(self, tasks: List[Task], description=None):
        super().__init__(tasks, description=description)

    @property
    def done(self):
        return super().done
