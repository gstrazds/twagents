import os
import logging
from symbolic.game import GameInstance
from symbolic import gv
from symbolic.decision_modules import Idler, Examiner, Interactor, Navigator, Hoarder #, YesNo, YouHaveTo, Darkness
from symbolic.decision_modules import GTNavigator
# from symbolic.knowledge_graph import *
from symbolic.event import NewTransitionEvent
from symbolic.entity import Entity
from symbolic.location import Location
from symbolic import knowledge_graph
from symbolic.action import GoNorth, GoSouth, GoEast, GoWest
# from twutils.twlogic import DIRECTION_RELATIONS

DIRECTION_ACTIONS = {
        'north_of': GoNorth,
        'south_of': GoSouth,
        'east_of': GoEast,
        'west_of': GoWest}

LOCATION_RELATIONS = ['at', 'in', 'on']


def find_door(fact_list, from_room, to_room):  # return name of door
    for fact in fact_list:
        if fact.name == 'link'\
        and fact.arguments[0].name == from_room.name \
        and fact.arguments[2].name == to_room.name:
            return fact.arguments[1].name
    return None


class NailAgent():
    """
    NAIL Agent: Navigate, Acquire, Interact, Learn

    NAIL has a set of decision modules which compete for control over low-level
    actions. Changes in world-state and knowledge_graph stream events to the
    decision modules. The modules then update how eager they are to take control.

    """
    def __init__(self, seed, env, rom_name, output_subdir='.'):
        self.setup_logging(rom_name, output_subdir)
        gv.rng.seed(seed)
        gv.dbg("RandomSeed: {}".format(seed))
        observed_knowledge_graph  = knowledge_graph.KnowledgeGraph()
        groundtruth_graph = knowledge_graph.KnowledgeGraph(groundtruth=True)
        self.gi = GameInstance(observed_knowledge_graph, groundtruth_graph)
        # self.knowledge_graph.__init__() # Re-initialize KnowledgeGraph
        # gv.event_stream.clear()
        self.gt_nav = GTNavigator(False)
        self.modules = [
                        self.gt_nav,
                        #Explorer(True),
                        # Navigator(True),
                        Idler(True),
                        Examiner(True), Hoarder(True),
                        # Interactor(True),
                        # YesNo(True), YouHaveTo(True), Darkness(True)
                        ]
        self.active_module    = None
        self.action_generator = None
        self.first_step       = True
        self._valid_detector  = None  #LearnedValidDetector()
        if env and rom_name:
            self.env = env
            self.step_num = 0

    def setup_logging(self, rom_name, output_subdir):
        """ Configure the logging facilities. """
        for handler in logging.root.handlers[:]:
            handler.close()
            logging.root.removeHandler(handler)
        self.logpath = os.path.join(output_subdir, 'nail_logs')
        if not os.path.exists(self.logpath):
            os.mkdir(self.logpath)
        self.kgs_dir_path = os.path.join(output_subdir, 'kgs')
        if not os.path.exists(self.kgs_dir_path):
            os.mkdir(self.kgs_dir_path)
        self.logpath = os.path.join(self.logpath, rom_name)
        logging.basicConfig(format='%(message)s', filename=self.logpath+'.log',
                            level=logging.DEBUG, filemode='w')

    def elect_new_active_module(self):
        """ Selects the most eager module to take control. """
        most_eager = 0.
        for module in self.modules:
            eagerness = module.get_eagerness(self.gi)
            if eagerness >= most_eager:
                self.active_module = module
                most_eager = eagerness
        print("elect_new_active_module:", "[NAIL](elect): {} Eagerness: {}".format(type(self.active_module).__name__, most_eager))
        gv.dbg("[NAIL](elect): {} Eagerness: {}".format(type(self.active_module).__name__, most_eager))
        self.action_generator = self.active_module.take_control(self.gi)
        self.action_generator.send(None)  # handshake with initial argless yield

    def generate_next_action(self, observation):
        """Returns the action selected by the current active module and
        selects a new active module if the current one is finished.

        """
        next_action = None
        while not next_action:
            try:
                next_action = self.action_generator.send(observation)
                print("[NAIL] (generate_next_action): {} {}".format(type(self.active_module).__name__, next_action))
            except StopIteration:
                self.consume_event_stream()
                self.elect_new_active_module()
        return next_action.text()

    def consume_event_stream(self):
        """ Each module processes stored events then the stream is cleared. """
        for module in self.modules:
            module.process_event_stream(self.gi)
        self.gi.event_stream.clear()

    def _get_gt_location(self, roomname, create_if_notfound=False):
        locations = self.gi.gt.locations_with_name(roomname)
        if locations:
            assert len(locations) == 1
            return locations[0]
        elif create_if_notfound:
            new_loc = Location(roomname)
            ev = self.gi.gt.add_location(new_loc)
            # DISCARD NewlocationEvent -- self.gi.event_stream.push(ev)
            print("created new GT Location:", new_loc)
            return new_loc
        return None

    def _get_gt_entity(self, name, locations=None, entitytype=None, create_if_notfound=False):
        if create_if_notfound:
            assert locations is not None
        entities = set()
        if locations:
            for l in locations:
                for e in l.entities:
                    if e.has_name(name):
                        entities.add(e)
        else:
            entities = self.gi.gt.entities_with_name(name, entitytype=entitytype)
        if entities:
            if len(entities) == 1:
                return list(entities)[0]
            else:
                found = None
                for e in entities:
                    if e._type == entitytype:
                        found = e
                if found:
                    return found
        if create_if_notfound:
            new_entity = Entity(name, locations[0], type=entitytype)
            ev = locations[0].add_entity(new_entity)
            if len(locations) > 0:
                for l in locations[1:]:
                    l.add_entity(new_entity)
            # DISCARD NewEntityEvent -- self.gi.event_stream.push(ev)
            print("created new GT Entity:", new_entity)
            return new_entity
        return None

    def gt_navigate(self, roomname):
        dest_loc = self._get_gt_location(roomname, create_if_notfound=False)
        assert dest_loc is not None
        self.gt_nav.set_goal(dest_loc, self.gi)
        # self.elect_new_active_module()

    def set_ground_truth(self, gt_facts):
        # print("GROUND TRUTH")
        player_loc = None
        door_facts = [fact for fact in gt_facts if fact.name == 'link']
        for fact in gt_facts:
            if fact.name in LOCATION_RELATIONS or fact.name in DIRECTION_ACTIONS:
                a0 = fact.arguments[0]
                a1 = fact.arguments[1]
                if a0.type == 'P' and a1.type == 'r':
                    player_loc = self._get_gt_location(a1.name, create_if_notfound=True)
                elif a0.type == 'r' and a1.type == 'r':
                    # print('++CONNECTION:', fact)
                    loc0 = self._get_gt_location(a0.name, create_if_notfound=True)
                    loc1 = self._get_gt_location(a1.name, create_if_notfound=True)
                    door_name = find_door(gt_facts, a1, a0)
                    if door_name:
                        door = self._get_gt_entity(door_name, entitytype=gv.DOOR, locations=[loc1, loc0], create_if_notfound=True)
                    else:
                        door = None
                    new_connection = knowledge_graph.Connection(loc1, DIRECTION_ACTIONS[fact.name], loc0, doorway=door)
                    self.gi.gt.add_connection(new_connection, self.gi)  # does nothing if connection already present
                elif a0.type == 'd' and a1.type == 'r':
                    # print("\t\tset_ground_truth: TODO door fact -- ", fact)
                    pass
                else:
                    # print("--IGNORING:", fact)
                    pass
        if player_loc:
            if self.gi.gt.set_player_location(player_loc, self.gi):
                print("CHANGED GT player location:", player_loc)

    def take_action(self, observation, obs_facts=None, gt_facts=None):
        if gt_facts:
            self.set_ground_truth(gt_facts)
        if obs_facts:
            # world = World.from_facts(facts)
            # add obs_facts to our KnowledgeGraph (self.gi.kg)
            pass
        if self.env and getattr(self.env, 'get_player_location', None):
            # Add true locations to the .log file.
            loc = self.env.get_player_location()
            if loc and hasattr(loc, 'num') and hasattr(loc, 'name') and loc.num and loc.name:
                gv.dbg("[TRUE_LOC] {} \"{}\"".format(loc.num, loc.name))

            # Output a snapshot of the kg.
            # with open(os.path.join(self.kgs_dir_path, str(self.step_num) + '.kng'), 'w') as f:
            #     f.write(str(self.knowledge_graph)+'\n\n')
            # self.step_num += 1

        observation = observation.strip()
        if self.first_step:
            gv.dbg("[NAIL] {}".format(observation))
            self.first_step = False
            return 'look' # Do a look to get rid of intro text

        if not self.gi.kg.player_location:
            loc = Location(observation)
            ev = self.gi.kg.add_location(loc)
            self.gi.kg.set_player_location(loc, self.gi)
            self.gi.kg._init_loc = loc
            # self.gi.event_stream.push(ev)

        self.consume_event_stream()

        if not self.active_module:
            self.elect_new_active_module()

        next_action = self.generate_next_action(observation)
        return next_action

    def observe(self, prev_obs, action, score, new_obs, terminal):
        """ Observe will be used for learning from rewards. """
#        p_valid = self._valid_detector.action_valid(action, new_obs)
#        gv.dbg("[VALID] p={:.3f} {}".format(p_valid, clean(new_obs)))
#        if kg.player_location:
#            dbg("[EAGERNESS] {}".format(' '.join([str(module.get_eagerness()) for module in self.modules[:5]])))
        self.gi.event_stream.push(NewTransitionEvent(prev_obs, action, score, new_obs, terminal))
        self.gi.action_recognized(action, new_obs)  # Update the unrecognized words
        if terminal:
            self.gi.kg.reset(self.gi)

    def finalize(self):
        # with open(self.logpath+'.kng', 'w') as f:
        #     f.write(str(self.knowledge_graph)+'\n\n')
        pass
