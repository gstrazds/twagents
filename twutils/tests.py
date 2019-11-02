import unittest

from symbolic.task import *
from symbolic.task_modules import SingleActionTask, SequentialActionsTask
from symbolic.action import StandaloneAction
from symbolic.decision_modules import TaskExecutor
from symbolic.game import GameInstance
from symbolic .knowledge_graph import KnowledgeGraph


def sequential_actions_task(start_num=1, count=4):
    return SequentialActionsTask(actions=[StandaloneAction(f"action{i}") for i in range(start_num, start_num+count)])

def create_simple_tasks(start_num=1, count=4):
    return [SingleActionTask(act=StandaloneAction(f"action{i}")) for i in range(start_num, start_num+count)]

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
        all_satisf = t.check_preconditions(None)
        self.assertFalse(all_satisf)
        self.assertEqual(len(t.missing.required_inventory), 0)
        self.assertEqual(len(t.missing.required_objects), 0)
        self.assertEqual(len(t.missing.required_locations), 0)
        self.assertEqual(len(t.missing.required_tasks), 1)
        self.assertEqual(t.missing.required_tasks[0], subt2)

    def test_singleaction(self):
        action1 = StandaloneAction('action1')
        t = SingleActionTask(act=action1)
        gen1 = t.activate(None)
        print(gen1)
        act = t.get_next_action("nothing to see here", None)
        self.assertIs(act, action1)
        next = t.get_next_action("should be done now", None)
        self.assertIsNone(next)
        self.assertFalse(t.is_active)

    def test_sequential_actions(self):
        count = 4
        mt = sequential_actions_task(start_num=1, count=count)
        mt.activate(None)
        obsnum = 0
        while mt.is_active:
            obsnum += 1
            act = mt.get_next_action(f"obs:{obsnum}", None)
            if act:
                self.assertEqual(act.verb, f"action{obsnum}")
            else:
                self.assertEqual(obsnum, count+1)
        self.assertFalse(mt.is_active)
        self.assertTrue(mt.is_done)
        self.assertFalse(mt.has_failed)

    def test_sequential_tasks(self):
        tasklist = create_simple_tasks(start_num=1, count=4)
        t1,t2,t3,t4 = tasklist
        self.assertFalse(t1.is_done)
        self.assertFalse(t2.is_done)
        self.assertFalse(t3.is_done)
        self.assertFalse(t4.is_done)
        mt = SequentialTasks(tasklist)
        mt.activate(None)
        obsnum = 0
        while mt.is_active:
            obsnum += 1
            act = mt.get_next_action(f"obs:{obsnum}", None)
            if act:
                self.assertEqual(act.verb, f"action{obsnum}")
            else:
                self.assertEqual(obsnum, len(tasklist)+1)
        self.assertFalse(mt.is_active)
        self.assertFalse(mt.has_failed)
        self.assertTrue(mt.is_done)
        self.assertTrue(t1.is_done)
        self.assertFalse(t1.is_active)
        self.assertTrue(t2.is_done)
        self.assertFalse(t2.is_active)
        self.assertTrue(t3.is_done)
        self.assertFalse(t4.is_active)
        self.assertTrue(t4.is_done)
        self.assertFalse(t4.is_active)

    def test_sequential_nested(self):
        t0 = SingleActionTask(act=StandaloneAction("action0"))
        tasks1 = create_simple_tasks(start_num=1, count=4)
        t5 = SingleActionTask(act=StandaloneAction("action5"))
        t6 = SingleActionTask(act=StandaloneAction("action6"))
        tasks2 = create_simple_tasks(start_num=7, count=3)
        mt1 = SequentialTasks(tasks1)
        mt2 = SequentialTasks(tasks2)
        mt3 = sequential_actions_task(start_num=10, count=5)
        tasklist = [t0, mt1, t5, t6, mt2, mt3]
        mt = SequentialTasks(tasklist)
        mt.activate(None)
        obsnum = 0
        done_tasks = []
        while mt.is_active:
            # print("+++NESTED SEQ OBSNUM: ", obsnum)
            act = mt.get_next_action(f"obs:{obsnum}", None)
            if act:
                self.assertEqual(act.verb, f"action{obsnum}")
            else:
                self.assertEqual(obsnum, len(tasklist)+len(tasks1)+len(tasks2)+len(mt3.actions)-3)
            obsnum += 1
        self.assertFalse(mt.is_active)
        self.assertFalse(mt.has_failed)
        self.assertTrue(mt.is_done)
        self.assertTrue(mt1.is_done)
        self.assertFalse(mt1.is_active)
        self.assertFalse(mt.has_failed)
        self.assertTrue(mt2.is_done)
        self.assertFalse(mt2.is_active)
        self.assertFalse(mt2.has_failed)
        self.assertTrue(mt2.is_done)
        self.assertFalse(mt2.is_active)
        self.assertFalse(mt2.has_failed)
        for t in done_tasks:
            self.assertTrue(t.is_done)
            self.assertFalse(t.is_active)

    def test_taskexec(self):
        print("\n----- testing TaskExecutor------")
        gi = GameInstance()
        te = TaskExecutor()
        te.activate()
        self.assertTrue(te._active)
        action_gen = te.take_control(gi)
        action_gen.send(None)   #handshake: decision_module protocol
        for counter, act in enumerate(action_gen):
            print(counter, act)
        # te.deactivate()
        self.assertFalse(te._active)
        self.assertEqual(te.get_eagerness(gi), 0)

    def test_taskexec_act1(self):
        print("......... TaskExecutor with one SingleActionTask ...")
        gi = GameInstance()
        te = TaskExecutor()
        StandaloneAction('action1')
        t = SingleActionTask(act=StandaloneAction('act1'))
        te.push_task(t)
        te.activate()
        action_gen = te.take_control(gi)
        action_gen.send(None)   #handshake: decision_module protocol
        for counter, act in enumerate(action_gen):
            print(counter, act)
            self.assertEqual(counter, 0)
            self.assertEqual(act.verb, "act1")
        # te.deactivate()
        self.assertFalse(te._active)
        self.assertEqual(te.get_eagerness(gi), 0)

    def test_taskexec_queue(self):
        print("......... TaskExecutor queue 3 SingleActionTasks ...")
        gi = GameInstance()
        te = TaskExecutor()
        StandaloneAction('action1')
        t1 = SingleActionTask(act=StandaloneAction('act1'))
        t2 = SingleActionTask(act=StandaloneAction('act2'))
        t3 = SingleActionTask(act=StandaloneAction('act3'))
        te.queue_task(t1)
        te.queue_task(t2)
        te.queue_task(t3)
        te.activate()
        action_gen = te.take_control(gi)
        action_gen.send(None)  # handshake: decision_module protocol
        for counter, act in enumerate(action_gen):
            print(counter, act)
            if counter == 0:
                self.assertEqual(act.verb, "act1")
            elif counter == 1:
                self.assertEqual(act.verb, "act2")
            elif counter == 2:
                self.assertEqual(act.verb, "act3")
        self.assertEqual(counter, 2)
        # te.deactivate()
        self.assertFalse(te._active)
        self.assertEqual(te.get_eagerness(gi), 0)

if __name__ == '__main__':
    unittest.main()