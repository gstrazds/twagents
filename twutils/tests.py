import unittest

from symbolic.task import *


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
        self.assertFalse(p.done)
        for t in subtasks:
            t._done = True
        self.assertTrue(p.done)
        subtasks[1]._done = False
        self.assertTrue(p.done)
        p.reset()
        self.assertFalse(p.done)
        subtasks[1]._done = True
        self.assertTrue(p.done)
        p.reset_all()
        self.assertFalse(p.done)
        for t in subtasks:
            self.assertFalse(t.done, "reset_all() should also reset all subtasks")

    def test_composite_done(self):
        self._check_composite_done(ParallelTasks)
        self._check_composite_done(SequentialTasks)

if __name__ == '__main__':
    unittest.main()