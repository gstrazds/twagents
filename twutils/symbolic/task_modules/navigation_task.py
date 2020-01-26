from typing import List
from ..knowledge_graph import *
from ..action import *
from ..entity import Location
from ..task import Task, SequentialTasks
from .tasks import SequentialActionsTask
from ..gv import rng, dbg


_nav_actions = [GoNorth, GoSouth, GoWest, GoEast]
# NorthWest, SouthWest, NorthEast, SouthEast, Up, Down, Enter, Exit]


# def get_direction_from_navaction(navaction):
#     return navaction.verb

def get_door_if_closed(connection):
    # check state of doorway, returns None if no door or the door is already open...
    if connection.doorway and connection.doorway.state.openable and not connection.doorway.state.is_open:
        return connection.doorway
    return None

class ExploreHereTask(Task):
    def __init__(self, description='', use_groundtruth=False):
        super().__init__(description=description, use_groundtruth=use_groundtruth)
        # self.add_postcondition( check if here has any unopened containers or doors )

    def check_result(self, result: str, kg: KnowledgeGraph) -> bool:
        return True

    def _generate_actions(self, kg) -> Action:
        """ Generates a sequence of actions.
        :type kg: KnowledgeGraph
        """
        here = kg.player_location
        ignored = yield   # required handshake
        response = yield Look
        self.check_result(response, kg)
        for entity in list(here.entities):
            print(f"ExploreHereTask -- {here} {entity} is_container:{entity.is_container}")
            if entity.is_container and entity.state.openable:
                print(f"ExploreHereTask -- {entity}.is_open:{entity.state.is_open}")
                if entity.state.openable and not entity.state.is_open:
                    response = yield Open(entity)
                    if self.check_result(response, kg):
                        entity.open()
        return None


class GoToNextRoomTask(Task):
    def __init__(self, connection=None, description='', use_groundtruth=False):
        super().__init__(description=description, use_groundtruth=use_groundtruth)
        assert connection
        self._connection = connection

    def check_result(self, response: str, kg: KnowledgeGraph) -> bool:
        return True

    def verify_opened(self, response: str, kg: KnowledgeGraph) -> bool:
        return "open" in response

    def _generate_actions(self, kg) -> Action:
        """ Generates a sequence of actions.
        :type kg: KnowledgeGraph
        """
        here = kg.player_location
        link = self._connection
        ignored = yield   # required handshake
        if link.from_location != here:
            print(f"{self} ASSERTION FAILURE from_location={link.from_location} != player_location={here}")
            self._failed = True
            self.deactivate(kg)
        elif link:
            door = get_door_if_closed(link)
            if door:
                response = yield Open(door)
                # check response to verify that door has been opened
                self.verify_opened(response, kg)
            response = yield link.action
            if not self.check_result(response, kg):
                self._failed = True
        return None


class PathTask(SequentialTasks):
    """
    Responsible for sequencing navigation actions to
    take the agent to a given destination, using the knowledge graph.
    """
    def __init__(self, goalname: str, description=None, use_groundtruth=True):
        self.goal_name = goalname
        self.goal_location = None
        self.path = None    # shortest path from start location to goal_location
        task_list: List[Task] = []
        if not description:
            description = f"PathTask[{goalname}]"
        super().__init__(tasks=task_list, description=None, use_groundtruth=False)

    def activate(self, kg, exec):
        print("PathTask.activate!!!!")
        if self.set_goal(kg) and self.path:  # might auto self.deactivate(kg)
            return super().activate(kg, exec)
        else:
            self.deactivate(kg)
            return None

    @property
    def maybe_GT(self):
        return "GT " if self.use_groundtruth else ""

    def set_goal(self, kg: KnowledgeGraph) -> bool:
        print("PathTask.set_goal()", self.goal_name)
        failure = False   # return value: initially optimistic
        assert self.goal_name
        if self.goal_name == '+NearestUnexplored+':
            self.goal_location = None
            self.path = kg.path_to_unknown()
        else:
            self.goal_location = kg.location_of_entity_with_name(self.goal_name)
            print(f"PathTask {self} unknown location or goal entity: {self.goal_name}")
            if not self.goal_location:
                self._failed = True
                self.deactivate(kg)
                return False

            # compute shortest path from current location to goal
            current_loc = kg.player_location
            self.path = kg.connections.shortest_path(current_loc, self.goal_location)
        print("{}NavigationTask set_goal({}) => {}".format(self.maybe_GT, self.goal_name, self.path))

        if self.path:
            link_desc = ','.join(["goto({})".format(link.to_location.name) for link in self.path])
            self.description = f"NavigationTask({self.goal_name})[{link_desc}]"
            self.tasks = [GoToNextRoomTask(link) for link in self.path]
        else:
            if self.goal_location == current_loc:
                self._done = True
                # gv.dbg(
                print("[{}PathTask] NOOP: already at goal {}={}".format(
                    self.maybe_GT, self.goal_location, current_loc))
            else:
                errmsg = "[{}PathTask] NOPATH: (current={}, goal={}) path={}".format(
                    self.maybe_GT, current_loc, self.goal_name, self.path)
                gv.dbg(errmsg)
                self.goal_location = None
                self._done = True
                failure = True
                print(errmsg + "\n" +
                      "   +++++++++++++++++   Connection Graph   ++++++++++++++++ \n" +
                      kg.connections.to_string() + "\n" +
                      "   +++++++++++++++++   +++++++++++++++   +++++++++++++++++")
            self._failed = failure
            self.deactivate(kg)
        return not failure


class FindTask(Task):
    def __init__(self, objname=None, description='', use_groundtruth=False):

        def _location_of_obj_is_known(kgraph): #closure (captures self) for postcondition check
            retval = kgraph.location_of_entity_is_known(self._objname)
            print(f"POSTCONDITION location_of_obj_is_known({self._objname}) => {retval}")
            return retval

        assert objname
        self._objname = objname
        if not description:
            description = f"FindTask[{objname}]"
        super().__init__(description=description, use_groundtruth=use_groundtruth)
        self.add_postcondition(_location_of_obj_is_known)

    # def check_result(self, response: str, kg: KnowledgeGraph) -> bool:
    #     return True

    def _generate_actions(self, kg) -> Action:
        """ Generates a sequence of actions.
        :type kg: KnowledgeGraph
        """
        ignored = yield   # required handshake
        here = kg.player_location

        # TODO: more code here....
        while not kg.location_of_entity_is_known(self._objname):
            if self._children:
                pathtask = self._children[0]
                assert isinstance(pathtask, PathTask)
                if pathtask.has_failed:
                    self._failed = True
                    break
                if pathtask.is_done and not pathtask.has_failed:
                    pathtask.reset_all()
            else:
                pathtask = PathTask('+NearestUnexplored+', use_groundtruth=self.use_groundtruth)
                self._children.append(pathtask)
            if not pathtask.has_failed:
                self._task_exec.start_prereq_task(pathtask)
        return None


class GoToTask(Task):
    def __init__(self, objname=None, description='', use_groundtruth=False):
        assert objname
        self._objname = objname
        if not description:
            description = f"FindTask[{objname}]"
        super().__init__(description=description, use_groundtruth=use_groundtruth)
        self.prereq.add_required_task(FindTask(objname, use_groundtruth=use_groundtruth))

    def _generate_actions(self, kg) -> Action:
        """ Generates a sequence of actions.
        :type kg: KnowledgeGraph
        """
        ignored = yield   # required handshake
        here = kg.player_location

        # TODO: more code here....
        if not kg.location_of_entity_is_known(self._objname):
            self.activate_subtask( PathTask('+NearestUnexplored+') )
        return None


class Foo:
    def get_path_to_goal_by_name(self, gi) -> bool:
        current_loc = self._knowledge_graph(gi).player_location
        print(f"{self.maybe_GT}GoToTask.set_goal_by_name({self._goal_name} (player_location={current_loc})")
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
                print(f"{self.maybe_GT}GoToTask({self._goal_name},{self.goal_location}) ignoring event NeedToGoTo:{event.target_location} GT={event.is_groundtruth}")
        elif isinstance(event, GroundTruthComplete):
            if self._goal_name and not self._active:
                name_of_goal = self._goal_name
                self.set_goal_by_name(self._goal_name, gi)
                if not self._active:
                    print(f"FAILED to set_goal_by_name({name_of_goal}) (no UnknownLocations?) -> CANCELLED!")
                    self._goal_name = None


    def get_next_action(self, kg):
        """
            Take next step along the shortest path to goal location
        """
        current_loc = kg.player_location
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
                    self.set_goal_by_name(self._goal_name, kg)
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
            assert next_step.from_location == kg.player_location
        # direction = get_direction_from_navaction(next_step.action)

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

    def get_door_if_closed(connection):
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
                    print(f"GoToTask -- {entity}.is_open:{entity.state.is_open}")
                    if entity.state.openable and not entity.state.is_open:
                        response = yield Open(entity)
                        entity.open()

            act = self.get_next_action(self._knowledge_graph(gi))
            response = yield act
            # TODO: should check response to verify that actions are succeeding
            if not act:
                break
