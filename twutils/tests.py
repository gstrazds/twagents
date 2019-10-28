import unittest

from symbolic.task import *
from symbolic.task_modules import SingleActionTask
from symbolic.action import StandaloneAction

action1 = StandaloneAction('action1')
action2 = StandaloneAction('action2')
action3 = StandaloneAction('action3')
action4 = StandaloneAction('action4')

class TestTask(unittest.TestCase):
    def test_list_int(self):
        """
        Test that it can sum a list of integers
        """
        data = [1, 2, 3]
        result = sum(data)
        self.assertEqual(result, 6)

    def test_create(self):
        t = Task("EmptyTask")
        self.assertEqual(str(t), "EmptyTask")

    def test_create_parallel(self):
        subtasks = [Task("Task1"), Task("Task2"), Task("Task3")]
        t = ParallelTasks(subtasks)
        self.assertEqual(len(t.tasks), 3)
        self.assertEqual(t.description, "ParallelTasks[Task1, Task2, Task3]")

    def test_create_sequential(self):
        subtasks = [Task("Task1"), Task("Task2"), Task("Task3")]
        t = SequentialTasks(subtasks)
        self.assertEqual(len(t.tasks), 3)
        self.assertEqual(t.description, "SequentialTasks[Task1, Task2, Task3]")

    def _check_composite_done(self, cls):
        subtasks = [Task("Task1"), Task("Task2"), Task("Task3")]
        p = cls(subtasks)
        self.assertEqual(len(p.tasks), 3)
        self.assertFalse(p.is_done)
        for t in subtasks:
            t._done = True
        self.assertTrue(p.is_done)
        subtasks[1]._done = False
        self.assertTrue(p.is_done)
        p.reset()
        self.assertFalse(p.is_done)
        subtasks[1]._done = True
        self.assertTrue(p.is_done)
        p.reset_all()
        self.assertFalse(p.is_done)
        for t in subtasks:
            self.assertFalse(t.is_done, "reset_all() should also reset all subtasks")

    def test_composite_done(self):
        self._check_composite_done(ParallelTasks)
        self._check_composite_done(SequentialTasks)

    def test_prereqs1(self):
        t = Task("Task1")
        subt1 = Task("Task_sub1")
        subt2 = Task("Task_sub2")
        t.prereq.required_tasks += [subt1, subt2]
        subt1._done = True
        all_satisf, missing = t.check_preconditions(None)
        self.assertFalse(all_satisf)
        self.assertEqual(len(missing.required_inventory), 0)
        self.assertEqual(len(missing.required_objects), 0)
        self.assertEqual(len(missing.required_locations), 0)
        self.assertEqual(len(missing.required_tasks), 1)
        self.assertEqual(missing.required_tasks[0], subt2)

    def test_singleaction(self):
        t = SingleActionTask(act=action1)
        gen1 = t.activate(None)
        print(gen1)
        act = t.get_next_action("nothing to see here", None)
        self.assertIs(act, action1)
        next = t.get_next_action("should be done now", None)
        self.assertIsNone(next)
        self.assertFalse(t.is_active)


if __name__ == '__main__':
    unittest.main()