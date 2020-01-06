from ..decision_module import DecisionModule
from ..knowledge_graph import *
from ..action import *
from ..entity import Location
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
    def __init__(self, active=False, use_groundtruth=True):
        super().__init__()
        self._debug = False
        self.use_groundtruth = use_groundtruth
        self._active = active
        self._nav_actions = [GoNorth, GoSouth, GoWest, GoEast]
            # NorthWest, SouthWest, NorthEast, SouthEast, Up, Down, Enter, Exit]
        self._suggested_directions = []
        self._active_eagerness = 1.0
        self._low_eagerness = 0.0
        self._goal_name = None
        self.goal_location = None
        self.path = None    # shortest path from start location to goal_location
        self._path_idx = -1  # next step along path
        self._opened_door = None  # state flag for keeping track of Open(door) actions
        self._close_door = None  # remember that we want to close a door that we opened

    @property
    def maybe_GT(self):
        return "GT " if self.use_groundtruth else ""

    def _knowledge_graph(self, gi):
        if self.use_groundtruth:
            return gi.gt
        return gi.kg

    def set_goal(self, goal: Location, gi: GameInstance):
        self.goal_location = goal
        # compute shortest path from current location to goal
        current_loc = self._knowledge_graph(gi).player_location
        self.path = self._knowledge_graph(gi).connections.shortest_path(current_loc, goal)
        print("{}Navigator set_goal({}) => {}".format(self.maybe_GT, goal, self.path))
        if self.path:
            self._active = True
            self._path_idx = 0
        else:
            print("   +++++++++++++++++   Connection Graph   ++++++++++++++++")
            print(self._knowledge_graph(gi).connections.to_string())
            print("   +++++++++++++++++   +++++++++++++++   +++++++++++++++++")
            if self.goal_location == current_loc:
                gv.dbg("[{}NAV] NOOP: already at goal {}={}".format(
                    self.maybe_GT, self.goal_location, current_loc))
            else:
                gv.dbg("[{}NAV] NOPATH: (current={}, goal={}) path={}".format(
                    self.maybe_GT, current_loc, self.goal_location, self.path))
            self._active = False
            self._path_idx = -1
        # adjust eagerness

    def set_goal_by_name(self, goal_name, gi):
        if self.use_groundtruth:
            locs = gi.gt.locations_with_name(goal_name)
        else:
            locs = gi.kg.locations_with_name(goal_name)
        if locs:
            self.set_goal(locs[0], gi)
            if self._active:
                return   # set_goal successful
        # loc not known or path to loc not found
        self._goal_name = goal_name
        current_loc = self._knowledge_graph(gi).player_location
        print(f"{self.maybe_GT} Navigator.set_goal_by_name({self._goal_name} (player_location={current_loc})")
        self.set_goal(self._knowledge_graph(gi)._unknown_location, gi)
        # if current_loc != self._knowledge_graph(gi)._unknown_location:
        #     self.set_goal(self._knowledge_graph(gi)._unknown_location, gi)

    def process_event(self, event, gi: GameInstance):
        """ Process an event from the event stream. """
        if isinstance(event, NeedToGoTo):
            self._goal_name = None
            self.set_goal_by_name(event.target_location, gi)
        elif isinstance(event, GroundTruthComplete):
            if self._goal_name and not self._active:
                self.set_goal(self._knowledge_graph(gi)._unknown_location, gi)

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
            current_loc = self._knowledge_graph(gi).player_location
            if current_loc == self.goal_location or current_loc.name == self._goal_name:
                gv.dbg("[{}NAV] Goal reached: {}".format(
                    self.maybe_GT, self.goal_location))
                if current_loc.name == self._goal_name or not self._goal_name:
                    self._goal_name = None
                    self._active = False
                elif self._goal_name:
                    self.set_goal_by_name(self._goal_name)
            else:
                gv.dbg("[{}NAV] FAILED! {} {} {}".format(
                    self.maybe_GT, self.goal_location, self._path_idx, self.path))
                if self._goal_name:
                    self.set_goal_by_name(self._goal_name, gi)
                else:
                    self._active = False
            if self._debug and not self._active:
                print("{}Navigator resetting _active=False", self.maybe_GT, self._path_idx, self.path)
            if self._close_door:
                close_door = Close(self._close_door)
                self._close_door = None
                # return close_door
            return None
        next_step = self.path[self._path_idx]
        # TODO: check for closed door -> "open <the door>" and don't increment _path_idx
        if self._debug:
            assert next_step.from_location == self._knowledge_graph(gi).player_location
        direction = get_direction_from_navaction(next_step.action)
        door = self.get_door_if_closed(next_step)
    #TODO: fix get_door_if_closed() to check for closed(door)...
        if door is not None and not self._opened_door:
            #Temporarily revert: don't close doors that we open
            self._opened_door = door
            return Open(door)
        if self._close_door:
            close_door = Close(self._close_door)
            self._close_door = None
            # return close_door
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
        if connection.doorway and connection.doorway.state.openable and not connection.doorway.state.is_open:
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
