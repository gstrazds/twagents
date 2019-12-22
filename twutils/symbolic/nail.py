import os
import logging
# from symbolic.game import GameInstance
# from symbolic import gv
# from symbolic.decision_modules import Idler, Examiner, Interactor, Navigator, Hoarder #, YesNo, YouHaveTo, Darkness
from symbolic.decision_modules import GTNavigator, GTEnder, GTRecipeReader, GTAcquire, TaskExecutor
from symbolic.event import NewTransitionEvent, GroundTruthComplete
# from symbolic.entity import Entity
from symbolic.location import Location
from symbolic.knowledge_graph import KnowledgeGraph
from symbolic.action import *
# from twutils.twlogic import DIRECTION_RELATIONS

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
    def __init__(self, seed, rom_name, env_name, output_subdir='.'):
        self.setup_logging(env_name, output_subdir)
        gv.rng.seed(seed)
        gv.dbg("RandomSeed: {}".format(seed))
        observed_knowledge_graph = KnowledgeGraph(None, groundtruth=False)
        groundtruth_graph = KnowledgeGraph(None, groundtruth=True)
        self.gi = GameInstance(kg=observed_knowledge_graph, gt=groundtruth_graph)
        # self.knowledge_graph.__init__() # Re-initialize KnowledgeGraph
        # gv.event_stream.clear()
        self.gt_nav = GTNavigator(False)
        self.modules = [
                        TaskExecutor(True),
                        # GTEnder(True),
                        self.gt_nav,
                        GTRecipeReader(),
                        GTAcquire(True),
                        #Explorer(True),
                        # Navigator(True),
                        #Idler(True),
                        #Examiner(True),
                        #Hoarder(True),
                        # Interactor(True),
                        # YesNo(True), YouHaveTo(True), Darkness(True)
                        ]
        self.active_module    = None
        self.action_generator = None
        self.first_step       = True
        self._valid_detector  = None  #LearnedValidDetector()
        if env_name and rom_name:
            self.rom_name = rom_name
            self.env_name = env_name
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
        failed_counter = 0
        while not next_action:
            failed_counter += 1
            if failed_counter > 10:
                gv.dbg(f"[NAIL] generate_next_action FAILED {failed_counter} times! BREAKING LOOP => |Look|")
                return "Look"
            try:
                next_action = self.action_generator.send(observation)
                print("[NAIL] (generate_next_action): ({}) -> |{}|".format(type(self.active_module).__name__, next_action))
            except StopIteration:
                self.consume_event_stream()
                self.elect_new_active_module()
        return next_action.text()

    def consume_event_stream(self):
        """ Each module processes stored events then the stream is cleared. """
        for module in self.modules:
            module.process_event_stream(self.gi)
        self.gi.event_stream.clear()

    def take_action(self, observation, obs_facts=None):
        if obs_facts:
            # world = World.from_facts(facts)
            # add obs_facts to our KnowledgeGraph (self.gi.kg)
            self.gi.kg.add_facts(obs_facts, self.gi)

        if hasattr(self, 'env') and getattr(self.env, 'get_player_location', None):
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
            #GVS# return 'look' # Do a look to get rid of intro text

        if not self.gi.kg.player_location:
            loc = Location(description=observation)
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

    # def gt_navigate(self, roomname):
    #     dest_loc = self.gi.gt.get_location(roomname, self.gi, create_if_notfound=False)
    #     assert dest_loc is not None
    #     self.gt_nav.set_goal(dest_loc, self.gi)
    #     # self.elect_new_active_module()

    def set_ground_truth(self, gt_facts):
        # print("GROUND TRUTH")
        # sort into separate lists to control the order in which facts get processed
        # Reinitialize, build complete GT KG from scratch each time
        self.gi._set_knowledge_graph(KnowledgeGraph(None, groundtruth=True), groundtruth=True)
        self.gi.gt.add_facts(gt_facts, self.gi)
        self.gi.event_stream.push(GroundTruthComplete(groundtruth=True))



