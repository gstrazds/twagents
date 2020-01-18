from typing import List
from ..knowledge_graph import *
from ..action import *
from ..entity import Location
from ..task import Task
from .tasks import SequentialActionsTask
from ..gv import rng, dbg


_nav_actions = [GoNorth, GoSouth, GoWest, GoEast]
# NorthWest, SouthWest, NorthEast, SouthEast, Up, Down, Enter, Exit]


def get_direction_from_navaction(navaction):
    return navaction.verb


class NavigationTask(SequentialActionsTask):
    """
    Responsible for choosing navigation actions to
    take the agent to a given destination, using the knowledge graph.
    """
    def __init__(self, goalname: str, description=None, use_groundtruth=True, close_doors=False):
        super().__init__(actions=None, description=description, use_groundtruth=use_groundtruth)
        ### a bit of a hack here, bypassing super().__init__() and duplicating some of its code
        # self.actions = []
        # self._current_idx = -1
        # if not description:
        #     description = f"NavigationTask({goalname})"
        # else:
        #     description = description
        # Task.__init__(description=description, use_groundtruth=use_groundtruth)
        ### ----- end SequentialActionsTask.__init__()
        self._goal_name = goalname
        self.goal_location = None
        self.path = None    # shortest path from start location to goal_location
        self._path_idx = -1  # next step along path
        self._close_doors = close_doors
        self._opened_door = None  # state flag for keeping track of most recently opened door
        self._close_door = None  # remember that we want to close a door that we opened

    def _init_path(self, path_actions: List[Action], description=None, use_groundtruth=False):
        assert path_actions, "Required arg: must specify at least one Action"
        self.actions = path_actions
        self._current_idx = 0
        if not description:
            actions_desc = ','.join([str(act) for act in self.actions])
            self.description = f"NavigationTask({self._goal_name})[{actions_desc}]"
        else:
            description = description

    @property
    def maybe_GT(self):
        return "GT " if self.use_groundtruth else ""

    def _knowledge_graph(self, gi):
        if self.use_groundtruth:
            return gi.gt
        return gi.kg

    def set_goal(self, goal: Location, gi: GameInstance) -> bool:
        self.goal_location = goal
        # compute shortest path from current location to goal
        current_loc = self._knowledge_graph(gi).player_location
        self.path = self._knowledge_graph(gi).connections.shortest_path(current_loc, goal)
        print("{}NavigationTask set_goal({}) => {}".format(self.maybe_GT, goal, self.path))
        if self.path:
            self._active = True
            self._path_idx = 0
            return True
        else:
            print("   +++++++++++++++++   Connection Graph   ++++++++++++++++")
            print(self._knowledge_graph(gi).connections.to_string())
            print("   +++++++++++++++++   +++++++++++++++   +++++++++++++++++")
            self._active = False
            self._path_idx = -1
            if self.goal_location == current_loc:
                self._done = True
                gv.dbg("[{}NAV] NOOP: already at goal {}={}".format(
                    self.maybe_GT, self.goal_location, current_loc))
                return True
            else:
                gv.dbg("[{}NAV] NOPATH: (current={}, goal={}) path={}".format(
                    self.maybe_GT, current_loc, self.goal_location, self.path))
                self.goal_location = None
                self._done = True
                self._failed = True
                return False

    def get_path_to_goal_by_name(self, gi) -> bool:
        current_loc = self._knowledge_graph(gi).player_location
        print(f"{self.maybe_GT}NavigationTask.set_goal_by_name({self._goal_name} (player_location={current_loc})")
        if self.use_groundtruth:
            loc = gi.gt.location_of_entity_with_name(self._goal_name)
        else:
            loc = gi.kg.location_of_entity_with_name(self._goal_name)
        if loc:
            return self.set_goal(loc, gi)
        else:
            return self.set_goal(self._knowledge_graph(gi)._unknown_location, gi)

    def process_event(self, event, gi: GameInstance):
        """ Process an event from the event stream. """
        if isinstance(event, NeedToGoTo):
            if self.use_groundtruth == event.is_groundtruth and \
                    (not self.goal_location and not self._goal_name) and \
                    (not event.target_location == self._knowledge_graph(gi)._unknown_location.name):
                print((f"{self.maybe_GT}Navigator NeedToGoTo({event.target_location}:"))
                self.set_goal_by_name(event.target_location, gi)
                if not self._active:
                    name_of_goal = event.target_location
                    print(f"NeedToGoTo FAILED to set_goal_by_name({name_of_goal}) (no UnknownLocations?) -> CANCELLED!")
                    if name_of_goal.startswith("TryToFind("):
                        objname = name_of_goal[len("TryToFind("):-1]
                        gi.event_stream.push(NoLongerNeed([objname], groundtruth=self.use_groundtruth))
                        self._goal_name = None
            else:
                print(f"{self.maybe_GT}NavigationTask({self._goal_name},{self.goal_location}) ignoring event NeedToGoTo:{event.target_location} GT={event.is_groundtruth}")
        elif isinstance(event, GroundTruthComplete):
            if self._goal_name and not self._active:
                name_of_goal = self._goal_name
                self.set_goal_by_name(self._goal_name, gi)
                if not self._active:
                    print(f"FAILED to set_goal_by_name({name_of_goal}) (no UnknownLocations?) -> CANCELLED!")
                    self._goal_name = None


    def get_next_action(self, gi: GameInstance):
        """
            Take next step along the shortest path to goal location
        """
        current_loc = self._knowledge_graph(gi).player_location
        if not self.path or self._path_idx < 0 or self._path_idx >= len(self.path):
            if current_loc == self.goal_location or current_loc.name == self._goal_name:
                gv.dbg("[{}NAV] Goal reached: {}".format(self.maybe_GT, self.goal_location))
                if current_loc.name == self._goal_name or not self._goal_name:
                    self._goal_name = None
                    self._active = False
                    self.goal_location = None
                elif self._goal_name:
                    self.set_goal_by_name(self._goal_name)
            else:
                gv.dbg(f"[{self.maybe_GT}NAV] FAILED! {self.goal_location} {self._path_idx} {self.path}")
                if self._goal_name and not (self.goal_location and self.goal_location.name == self._goal_name):
                    self.set_goal_by_name(self._goal_name, gi)
                else:
                    self._goal_name = None
                    self._active = False
                    self.goal_location = None
            if self._close_door:
                close_door = Close(self._close_door)
                self._close_door = None
                # return close_door
            return None
        next_step = self.path[self._path_idx]
        if self._debug:
            assert next_step.from_location == self._knowledge_graph(gi).player_location
        direction = get_direction_from_navaction(next_step.action)

        # check for closed door -> "open <the door>" and don't increment _path_idx
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
        # check state of doorway, returns None if no door or the door is already open...
        if connection.doorway and connection.doorway.state.openable and not connection.doorway.state.is_open:
            return connection.doorway
        return None

    def take_control(self, gi: GameInstance):
        """
        Takes a navigation action and records the resulting transition.
        """
        obs = yield
        while True:
            current_loc = self._knowledge_graph(gi).player_location
            ## if self.open_all_containers:
            for entity in list(current_loc.entities):
                # print(f"GTNavigator -- {current_loc} {entity} is_container:{entity.is_container}")
                if entity.is_container and entity.state.openable:
                    print(f"NavigationTask -- {entity}.is_open:{entity.state.is_open}")
                    if entity.state.openable and not entity.state.is_open:
                        response = yield Open(entity)
                        entity.open()

            act = self.get_next_action(gi)
            response = yield act
            # TODO: should check response to verify that actions are succeeding
            if not act:
                break
