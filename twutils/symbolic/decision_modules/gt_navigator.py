from ..decision_module import DecisionModule
from ..knowledge_graph import *
from ..action import *
from ..location import Location
from ..gv import rng, dbg


def get_direction_from_navaction(navaction):
    return navaction.verb


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
        self._opened_door = None  # state flag for keeping track of Open(door) actions
        self._close_door = None  # remember that we want to close a door that we opened

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
            if self.goal_location == current_loc:
                gv.dbg("[GT NAV] NOOP: already at goal {}={}".format(self.goal_location, current_loc))
            else:
                gv.dbg("[GT NAV] NOPATH: (current={}, goal={}) path={}".format(
                                                            current_loc, self.goal_location, self.path))
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
            if gi.gt.player_location == self.goal_location:
                gv.dbg("[GT NAV] Goal reached: {}".format(self.goal_location))
            else:
                gv.dbg("[GT NAV] FAILED! {} {} {}".format(self.goal_location, self._path_idx, self.path))
            if self._debug:
                print("GT Navigator resetting _active=False", self._path_idx, self.path)
            return None
        next_step = self.path[self._path_idx]
        # TODO: check for closed door -> "open <the door>" and don't increment _path_idx
        if self._debug:
            assert next_step.from_location == gi.gt.player_location
        direction = get_direction_from_navaction(next_step.action)
        door = self.get_door_if_closed(next_step)
    #TODO: fix get_door_if_closed() to check for closed(door)...
        if door is not None and not self._opened_door:
            self._opened_door = door
            return Open(door)
        if self._close_door:
            close_door = Close(self._close_door)
            self._close_door = None
            return close_door
        self._path_idx += 1
        if self._opened_door:
            self._close_door = self._opened_door   #Close(self._opened_door)
            self._opened_door = None  # close the door on the next step (after going to dest room)
        return next_step.action

    def get_door_if_closed(self, connection):
        # rel = "{}_of".format(direction)
        # door_facts = world.state.facts_with_signature(Signature('link', ('r', 'd', 'r')))
        # for link in door_facts:
        #     if link.arguments[0].name == loc.name and link.arguments[2].name == dest.name:
        #         door = link.arguments[1]
        #         if world.state.is_fact(Proposition("closed", [door])):
        #             return entity_from_variable(door)

        #TODO: check state of doorway, return None if the door is already open...
        if connection.doorway and connection.doorway.state.openable() and not connection.doorway.state.is_open:
            return connection.doorway
        return None

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
