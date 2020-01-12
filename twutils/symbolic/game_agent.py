import os
import logging
# from symbolic.game import GameInstance
# from symbolic import gv
# from symbolic.decision_modules import Idler, Examiner, Interactor, Navigator, Hoarder #, YesNo, YouHaveTo, Darkness
from symbolic.decision_modules import GTNavigator, GTRecipeReader, GTAcquire, TaskExecutor   # GTEnder
from symbolic.event import GroundTruthComplete   # NewTransitionEvent
# from symbolic.entity import Entity
from symbolic.entity import Location
from symbolic.knowledge_graph import KnowledgeGraph
from symbolic.action import *
# from twutils.twlogic import DIRECTION_RELATIONS


class TextGameAgent:
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
        self.gt_nav = None
        self.modules = []
        self.active_module    = None
        self.action_generator = None
        self.first_step       = True
        self._valid_detector  = None  #LearnedValidDetector()
        self._last_action = None
        if env_name and rom_name:
            self.rom_name = rom_name
            self.env_name = env_name
            self.step_num = 0
        self._init_modules()

    def reset(self):
        groundtruth_graph = KnowledgeGraph(None, groundtruth=True)    # start fresh with empty graph
        observed_knowledge_graph = self.gi.kg   # reuse existing knowledge
        observed_knowledge_graph.reset()
        self.gi = GameInstance(kg=observed_knowledge_graph, gt=groundtruth_graph)
        self._init_modules()

    def _init_modules(self):
        self.gt_nav = GTNavigator(False, use_groundtruth=False)
        self.modules = [
                        TaskExecutor(True),
                        self.gt_nav,  # GTNavigator
                        GTAcquire(True, use_groundtruth=False),
                        # GTRecipeReader(use_groundtruth=False),
                        # Explorer(True),
                        # Navigator(True),
                        # Idler(True),
                        # Examiner(True),
                        # Hoarder(True),
                        # Interactor(True),
                        # YesNo(True), YouHaveTo(True), Darkness(True)
                        ]
        self.active_module    = None
        self.action_generator = None
        self._valid_detector  = None  #LearnedValidDetector()
        self.first_step       = True
        self._last_action = None

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

    def consume_event_stream(self):
        """ Each module processes stored events then the stream is cleared. """
        for module in self.modules:
            module.process_event_stream(self.gi)
        self.gi.event_stream.clear()

    def generate_next_action(self, observation):
        """Returns the action selected by the current active module and
        selects a new active module if the current one is finished.
        """
        next_action = None
        failed_counter = 0
        while not next_action:
            failed_counter += 1
            if failed_counter > 9:
                gv.dbg(f"[NAIL] generate_next_action FAILED {failed_counter} times! BREAKING LOOP => |Look|")
                # return "Look"
                next_action = Look
            else:
                try:
                    next_action = self.action_generator.send(observation)
                    print("[NAIL] (generate_next_action): ({}) -> |{}|".format(type(self.active_module).__name__, next_action))
                except StopIteration:
                    self.consume_event_stream()
                    self.elect_new_active_module()
        return next_action

    def choose_next_action(self, observation, observable_facts=None, prev_action=None):
        if prev_action and self._last_action:
            assert prev_action == self._last_action.text()

        if hasattr(self, 'env') and getattr(self.env, 'get_player_location', None):
            # Add true locations to the .log file.
            loc = self.env.get_player_location()
            if loc and hasattr(loc, 'num') and hasattr(loc, 'name') and loc.num and loc.name:
                gv.dbg("[TRUE_LOC] {} \"{}\"".format(loc.num, loc.name))
        # Output a snapshot of the kg.
        # with open(os.path.join(self.kgs_dir_path, str(self.step_num) + '.kng'), 'w') as f:
        #     f.write(str(self.knowledge_graph)+'\n\n')
        # self.step_num += 1

        if observable_facts and self.gi.kg:
            self.gi.kg.update_facts(observable_facts, prev_action=self._last_action)

        observation = observation.strip()
        if self.first_step:
            gv.dbg("[NAIL] {}".format(observation))
            self.first_step = False
            #GVS# return 'look' # Do a look to get rid of intro text

            if not self.gi.kg.player_location:
                assert False, "This code is no longer used"
                loc = Location(description=observation)
                ev = self.gi.kg.add_location(loc)
                self.gi.kg.set_player_location(loc)

        self.consume_event_stream()

        if not self.active_module:
            self.elect_new_active_module()

        next_action = self.generate_next_action(observation)
        self._last_action = next_action
        return next_action.text()

    def observe(self, prev_obs, prev_action, score, new_obs, terminal):
        """ Observe could be used for learning from rewards. """
#        p_valid = self._valid_detector.action_valid(action, new_obs)
#        gv.dbg("[VALID] p={:.3f} {}".format(p_valid, clean(new_obs)))

        # NewTransitionEvent unused by current code
        # self.gi.event_stream.push(NewTransitionEvent(prev_obs, prev_action, score, new_obs, terminal))
        if prev_action and self._last_action:
            assert prev_action == self._last_action.text()
        self.gi.action_recognized(prev_action, new_obs)  # Update the unrecognized words

    def finalize(self):
        # with open(self.logpath+'.kng', 'w') as f:
        #     f.write(str(self.knowledge_graph)+'\n\n')
        pass

    def set_ground_truth(self, gt_facts):
        # print("GROUND TRUTH")
        # Reinitialize, build complete GT KG from scratch each time
        self.gi.set_knowledge_graph(KnowledgeGraph(None, groundtruth=True), groundtruth=True)
        #TODO (Disabled ground truth)
        # self.gi.gt.update_facts(gt_facts)
        self.gi.event_stream.push(GroundTruthComplete(groundtruth=True))


