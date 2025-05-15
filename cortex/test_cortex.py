import unittest
import random

# class Dispatch(DfDecider):
#     def __init__(self):
#         super().__init__()

#         # self.add_child("flip_bottle", FlipBin())
#         self.add_child("pick_bottle", PickBottle())
#         self.add_child("place_bottle", PlaceBottle())
#         self.add_child("go_home", make_go_home())
#         self.add_child("do_nothing", DfStateMachineDecider(DoNothing()))

#     def decide(self):
#         if self.context.stack_complete:
#             return DfDecision("go_home")

#         if self.context.has_active_bottle:
#             if not self.context.active_bottle.is_attached:
#                 return DfDecision("pick_bottle")
#             # elif self.context.active_bottle.needs_flip:
#             #     return DfDecision("flip_bottle")
#             else:
#                 return DfDecision("place_bottle")
#         else:
#             return DfDecision("go_home")


# def make_decider_network(robot, monitor_fn):
#     grasp_target_marker = VisualCuboid(prim_path="/grasp_T",
#         position=np.array([0.5, 0.5, 0.0]),
#         scale=np.array([.15, .03, .04]),
#         color=np.array([.8, .2, 0.]),
#         visible=False)

class TestDeciderNetworks(unittest.TestCase):
    def test_load_module(self):
        """
        Test that df.py is loadable
        """
        from symbolic.cortex.df import (
            DfDecider,
            DfDecision,
            DfNetwork,
            DfSetLockState,
            DfState,
            DfStateMachineDecider,
            DfStateSequence,
            DfTimedDeciderState,
            DfWaitState,
            DfWriteContextState,
        )

        class S1(DfState):
            def __init__(self, my_val):
                self.my_val = my_val

            def step(self):
                if random.choice([1,2]) == 2:
                    return None
                return self

        s1 = S1(my_val=1)

        self.assertEqual(s1.my_val, 1)




