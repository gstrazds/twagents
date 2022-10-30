from typing import List
from ..knowledge_graph import *
from ..action import *
from ..task import Task, SequentialTasks

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
    def __init__(self, description='', use_groundtruth=False, look_first=False):
        super().__init__(description=description, use_groundtruth=use_groundtruth)
        self._verb = "explore"
        self._args = ['here']
        self.look_first = look_first
        # self._path_task = PathTask('NearestUnexplored', use_groundtruth=self.use_groundtruth)
        # self._children.append(self._path_task)
        # self.prereq.add_required_task(self._path_task)
        # self.add_postcondition( check if here has any unopened containers or doors )

    def check_result(self, result: str, kg: KnowledgeGraph) -> bool:
        return True

    def reset_all(self):
        self._path_task.reset_all()
        super().reset_all()

    def _generate_actions(self, kg) -> Action:
        """ Generates a sequence of actions.
        :type kg: KnowledgeGraph
        """
        here = kg.player_location
        ignored = yield   # required handshake
        if self.look_first:
            response = yield Look  #GVS NOTE: for TW envs, we request description infos, so Look is redundant
            self.check_result(response, kg)
        for entity in list(here.entities):
            # print(f"ExploreHereTask -- {here} {entity} is_container:{entity.is_container}")
            if entity.is_container and entity.state.openable:
                # print(f"ExploreHereTask -- {entity}.is_open:{entity.state.is_open}")
                if entity.state.openable and not entity.state.is_open:
                    response = yield Open(entity)
                    if self.check_result(response, kg):
                        entity.open()
        self._done = True
        return None


class GoToNextRoomTask(Task):
    def __init__(self, connection=None, description='', use_groundtruth=False):
        super().__init__(description=description, use_groundtruth=use_groundtruth)
        self._verb = "enter"
        assert connection
        self._connection = connection
        self._args = [connection.to_location.name]


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

        # ExploreHereTask
        ## if self.open_all_containers:
        # for entity in list(here.entities):
        #     # print(f"GoToNextRoomTask -- {here} {entity} is_container:{entity.is_container}")
        #     if entity.is_container and entity.state.openable:
        #         print(f"GoToNextRoomTask -- {entity}.is_open:{entity.state.is_open}")
        #         if entity.state.openable and not entity.state.is_open:
        #             response = yield Open(entity)
        #             entity.open()

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
        self._done = True
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
        super().__init__(tasks=task_list, description=None, use_groundtruth=use_groundtruth)
        self._verb = "walk_to"
        self._args = [goalname]

    def action_phrase(self) -> str:   # repr similar to a command/action (verb phrase) in the game env
        return f"{self._verb} {self.goal_name}"
        # words = [self._verb] + self._args
        # if len(words) > 2 and self._preposition:
        #     words.insert(2, self._preposition)
        # return " ".join(words)

    def activate(self, kg, exec):
        if self.is_active:
            print(f"{self} .activate(): ALREADY ACTIVE")
            return super().activate(kg, exec)
        else:
            # print("PathTask.activate!!!!")
            if self.set_goal(kg) and self.path:  # might auto self.deactivate(kg)
                return super().activate(kg, exec)
            else:
                # self._failed = True  # this happens in set_goal()
                self.deactivate(kg)
                self._done = True
                return None

    def reset_all(self):
        self.goal_location = None
        self.path = None
        self.tasks = []
        super().reset_all()

    @property
    def maybe_GT(self):
        return "GT " if self.use_groundtruth else ""

    @property
    def description(self) -> str:
        description = f"PathTask({self.goal_name})"
        if self.path:
            link_desc = ','.join(["goto({})".format(link.to_location.name) for link in self.path])
            description +=  f"[{link_desc}]"
        return description

    def set_goal(self, kg: KnowledgeGraph) -> bool:
        super().reset_all()
        # print("PathTask.set_goal()", self.goal_name)
        failure = False   # return value: initially optimistic
        assert self.goal_name
        current_loc = kg.player_location
        if self.goal_name == 'NearestUnexplored':
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
            self.path = kg.connections.shortest_path(current_loc, self.goal_location)
        print("{}PathTask set_goal({}) => {}".format(self.maybe_GT, self.goal_name, self.path))

        if self.path:
            self.tasks = []   #ExploreHereTask(use_groundtruth=self.use_groundtruth)]
            self.tasks += [GoToNextRoomTask(link) for link in self.path]
            if self.goal_name == 'NearestUnexplored':
                self.tasks.append(ExploreHereTask(use_groundtruth=self.use_groundtruth))
        else:
            if self.goal_location == current_loc:
                self._done = True
                # gv.dbg(
                print("[{}PathTask] NOOP: already at goal {}={}".format(
                    self.maybe_GT, self.goal_location, current_loc))
            else:
                errmsg = "[{}PathTask] NOPATH: (current={}, goal={}) path={}".format(
                    self.maybe_GT, current_loc, self.goal_name, self.path)
                kg.dbg(errmsg)
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
            # print(f"POSTCONDITION location_of_obj_is_known({self._objname}) => {retval}")
            return retval

        assert objname
        self._objname = objname
        if not description:
            description = f"FindTask[{objname}]"
        super().__init__(description=description, use_groundtruth=use_groundtruth)
        self._verb = "find"
        self._args = [objname]
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
        if not kg.location_of_entity_is_known(self._objname):
            if self._children:
                go_somewhere_new = self._children[0]
                assert isinstance(go_somewhere_new, PathTask)
                if go_somewhere_new.has_failed:
                    self._failed = True
                    # break
                if go_somewhere_new.is_done and not go_somewhere_new.has_failed:
                    go_somewhere_new.reset_all()
            else:
                task_list = [PathTask('NearestUnexplored', use_groundtruth=self.use_groundtruth),
                             ExploreHereTask(use_groundtruth=self.use_groundtruth)]
                go_somewhere_new = task_list[0]  #SequentialTasks(tasks=task_list, use_groundtruth=self.use_groundtruth)
                self._children.append(go_somewhere_new)
            if not go_somewhere_new.is_active:
                if go_somewhere_new.has_failed:
                    self._done = True
                else:
                    self._task_exec.start_prereq_task(go_somewhere_new, self)
        return None


class GoToTask(Task):
    def __init__(self, objname=None, description='', use_groundtruth=False):
        def _location_of_obj_is_here(kgraph):  # closure (captures self) for postcondition check
            loc = kgraph.location_of_entity_with_name(self._objname)
            retval = (loc == kgraph.player_location)
            # print(f"POSTCONDITION location_of_obj_is_here({self._objname}) => {retval}")
            return retval

        assert objname
        self._objname = objname
        if not description:
            description = f"GoToTask[{objname}]"
        super().__init__(description=description, use_groundtruth=use_groundtruth)
        self._verb = "go_to"
        self._args = [objname]
        self.prereq.add_required_task(FindTask(objname, use_groundtruth=use_groundtruth))
        self.add_postcondition(_location_of_obj_is_here)

    def _generate_actions(self, kg) -> Action:
        """ Generates a sequence of actions.
        :type kg: KnowledgeGraph
        """
        ignored = yield   # required handshake
        here = kg.player_location

        # TODO: more code here....
        if kg.location_of_entity_is_known(self._objname):
            if self._children:
                pathtask = self._children[0]
                assert isinstance(pathtask, PathTask)
                assert pathtask.goal_name == self._objname
                if pathtask.has_failed:
                    self._failed = True
                    # break
                if pathtask.is_done and not pathtask.has_failed:
                    pathtask.reset_all()
            else:
                pathtask = PathTask(self._objname, use_groundtruth=self.use_groundtruth)
                self._children.append(pathtask)
            if not pathtask.has_failed and not pathtask.is_active:
                self._task_exec.start_prereq_task(pathtask, self)
        self._done = True
        return None


