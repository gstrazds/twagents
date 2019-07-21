from ..decision_module import DecisionModule
from ..knowledge_graph import *
from ..action import *
from ..location import Location
from ..gv import rng, dbg

class GTNavigator(DecisionModule):
    """
    The Ground Truth Navigator is responsible for choosing navigation actions to
    take the agent to a given destination, using the Ground Truth knowledge graph.
    (A training and debugging aid, to generate trajectories for imitation learning
    or to bring the agent to specific initial states)

    Args:
    eagerness: Default eagerness for this module

    """
    def __init__(self, active=False):
        super().__init__()
        self._debug = False
        self._active = active
        self._nav_actions = [GoNorth, GoSouth, GoWest, GoEast]
            # NorthWest, SouthWest, NorthEast, SouthEast, Up, Down, Enter, Exit]
        self._suggested_directions = []
        self._active_eagerness = 1.0
        self._low_eagerness = 0.0
        self.goal_location = None
        self.path = None    # shortest path from start location to goal_location
        self._path_idx = -1  # next step along path

    def set_goal(self, goal: Location, gi: GameInstance):
        self.goal_location = goal
        # compute shortest path from current location to goal
        current_loc = gi.gt.player_location
        self.path = gi.gt.connections.shortest_path(current_loc, goal)
        print("GTNavigator set_goal({}) => {}".format(goal, self.path))
        if self.path:
            self._active = True
            self._path_idx = 0
        else:
            self._active = False
            self._path_idx = -1
        # adjust eagerness

    def process_event(self, event, gi: GameInstance):
        """ Process an event from the event stream. """
        pass

    def get_eagerness(self, gi: GameInstance):
        if not self._active:
            return self._low_eagerness
        else:
            return self._active_eagerness
        # if self.get_unexplored_actions(gi.kg.player_location, gi):
        #     return self._default_eagerness
        # return rng.choice([self._low_eagerness, self._default_eagerness])

    def get_next_action(self, gi: GameInstance):
        """
            Take next step along the shortest path to goal location
        """
        if not self.path or self._path_idx < 0 or self._path_idx >= len(self.path):
            self._active = False
            if self._debug:
                print("GT Navigator resetting _active=False", self._path_idx, self.path)
            return None
        next_step = self.path[self._path_idx]
        # TODO: check for closed door:  "open <the door>" and don't increment _path_idx
        self._path_idx += 1
        if self._debug:
            assert next_step.from_location == gi.gt.player_location
        return next_step.action

    def take_control(self, gi: GameInstance):
        """
        Takes a navigation action and records the resulting transition.

        """
        obs = yield
        while True:
            act = self.get_next_action(gi)
            response = yield act
            # TODO: could potentially check response to verify that actions are succeeding
            if not act:
                break
