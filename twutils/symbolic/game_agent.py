import os
import logging
import random

from symbolic.decision_modules import TaskExec
# from symbolic.event import GroundTruthComplete   # NewTransitionEvent
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
    def __init__(self, seed, rom_name, env_name, idx=0, game=None, output_subdir='.'):
        self._idx = idx
        self.setup_logging(env_name, idx, output_subdir)
        if game:
            self._game = game  # if provided, can do nicer logging
        self.dbg("RandomSeed: {}".format(seed))
        rng = random.Random()  # each game instance gets its own random number generator
        rng.seed(seed)
        observed_knowledge_graph = KnowledgeGraph(None, groundtruth=False, logger=self.get_logger())
        groundtruth_graph = KnowledgeGraph(None, groundtruth=True, logger=self.get_logger())
        self.gi = GameInstance(kg=observed_knowledge_graph, gt=groundtruth_graph, rng=rng, logger=self.get_logger())
        # self.knowledge_graph.__init__() # Re-initialize KnowledgeGraph
        # gv.event_stream.clear()
        self.task_exec = None
        self.modules = []
        self.active_module    = None
        self.action_generator = None
        self.first_step = True
        self.step_num = 0
        self._valid_detector  = None  #LearnedValidDetector()
        self._last_action = None
        if env_name:
            self.env_name = env_name
            # self.rom_name = rom_name
        self._init_modules()

    @property
    def last_action(self) -> str:
        return self._last_action.text() if self._last_action else None

    def reset(self, forget_everything=False):
        groundtruth_graph = KnowledgeGraph(None, groundtruth=True, logger=self.get_logger())    # start fresh with empty graph
        if forget_everything:
            observed_knowledge_graph = KnowledgeGraph(None, groundtruth=False, logger=self.get_logger())    # start fresh with empty graph
        else:
            observed_knowledge_graph = self.gi.kg   # reuse existing knowledge
            observed_knowledge_graph.reset()        # bet restore to initial state (start of episode)
        self.gi = GameInstance(kg=observed_knowledge_graph, gt=groundtruth_graph, logger=self.get_logger())
        self._init_modules()
        self.first_step = True
        self.step_num = 0

    def _init_modules(self):
        self.task_exec = TaskExec(True)
        self.modules = [
                        self.task_exec,
                        # GTNavigator(False, use_groundtruth=False),
                        # GTAcquire(True, use_groundtruth=False),
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

    def setup_logging(self, game_id, idx, output_subdir="."):
        """ Configure the logging facilities. """
        logging.basicConfig(format='%(message)s',
                            level=logging.DEBUG, filemode='a')
        env_name = f"{game_id}_{idx:02}"
        print("setup_logging:", env_name)
        # for handler in logging.root.handlers[:]:
        #     handler.close()
        #     logging.root.removeHandler(handler)
        logdir = os.path.join(output_subdir, 'nail_logs')
        if not os.path.exists(logdir):
            os.mkdir(logdir)
        # self.kgs_dir_path = os.path.join(output_subdir, 'kgs')
        # if not os.path.exists(self.kgs_dir_path):
        #     os.mkdir(self.kgs_dir_path)
        self.logpath = os.path.join(logdir, env_name)
        # logging.basicConfig(format='%(message)s', filename=self.logpath+'.log',
        #                     level=logging.DEBUG, filemode='w')
        self._logger_name = env_name
        logger = self.get_logger()
        logger.setLevel(logging.DEBUG)
        print("setup_logging - logger = ", logger)
        if logger.handlers:   # skip if this logger has already been configured
            print("setup_logging HAS HANDLERS = True!", logger.handlers[:], logging.root.handlers[:])
            for h in logger.handlers[:]:
                print(f"setup_logging({env_name}) PREEXISTING handler:", h)
        else:
            fh = logging.FileHandler(self.logpath, 'a')
            fh.setLevel(logging.DEBUG)
            # create console handler with a higher log level
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            ch.setFormatter(formatter)
            fh.setFormatter(formatter)
            logger.addHandler(fh)
            print("setup_logging ADDED HANDLER", fh)
            logger.addHandler(ch)
            print("setup_logging ADDED HANDLER", ch)
            for h in logger.handlers:
                print(f"setup_logging({env_name}) CONFIGURED NEW handler:", h)
        self.info("SETUP LOGGING -- env_name: "+env_name)
        self.dbg("TEST DBG LOGGING -- env_name: "+env_name)
        # self.warn("LOGGING: THIS SHOULD GET LOGGED!")

    def get_logger(self):
        return logging.getLogger(self._logger_name)

    def dbg(self, msg, *args, **kwargs):
        self.get_logger().debug(msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        self.get_logger().info(msg, *args, **kwargs)

    def warn(self, msg, *args, **kwargs):
        self.get_logger().warning(msg, *args, **kwargs)

    def elect_new_active_module(self):
        """ Selects the most eager module to take control. """
        most_eager = 0.
        for module in self.modules:
            eagerness = module.get_eagerness(self.gi)
            if eagerness >= most_eager:
                self.active_module = module
                most_eager = eagerness
        print("elect_new_active_module:", "[NAIL](elect): {} Eagerness: {}".format(type(self.active_module).__name__, most_eager))
        self.dbg("[NAIL](elect): {} Eagerness: {}".format(type(self.active_module).__name__, most_eager))
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
                msg = f"[game_agent] Step:{self.step_num} generate_next_action FAILED {failed_counter} times! BREAKING LOOP => |Look|"
                print(msg)
                self.dbg(msg)
                # return "Look"
                next_action = Look
            else:
                try:
                    next_action = self.action_generator.send(observation)
                    if next_action:
                        msg = f"[game_agent] Step:{self.step_num} (generate_next_action): ({type(self.active_module).__name__}) -> |{next_action}|"
                        self.dbg(msg)
                except StopIteration:
                    self.consume_event_stream()
                    self.elect_new_active_module()
        return next_action

    def choose_next_action(self, observation, observable_facts=None, prev_action=None):
        print("DEPRECATION WARNING: game_agent.choose_next_action()")
        self.update_kg(observation, observable_facts=observable_facts, prev_action=prev_action)
        return self.select_next_action(observation, external_next_action=None)

    def update_kg(self, observation, observable_facts=None, prev_action=None):
        if prev_action and self._last_action:
            if prev_action.lower() != self._last_action.text().lower() and \
                    not self._last_action.text().startswith("answer:"):
                print(f"WARNING: prev_action:|{prev_action}| != self._last_action: |{self._last_action.text()}|")
                # assert prev_action == self._last_action.text()
        if not prev_action:
            prev_action = self._last_action

        if hasattr(self, 'env') and getattr(self.env, 'get_player_location', None):
            # Add true locations to the .log file.
            loc = self.env.get_player_location()
            if loc and hasattr(loc, 'num') and hasattr(loc, 'name') and loc.num and loc.name:
                self.dbg("[TRUE_LOC] {} \"{}\"".format(loc.num, loc.name))
        # Output a snapshot of the kg.
        # with open(os.path.join(self.kgs_dir_path, str(self.step_num) + '.kng'), 'w') as f:
        #     f.write(str(self.knowledge_graph)+'\n\n')

        if observable_facts and self.gi.kg:
            self.gi.kg.update_facts(observable_facts, prev_action=prev_action)

        observation = observation.strip()
        if self.first_step:
            self.dbg("[game_agent] {}".format(observation))
            self.first_step = False
            # if not self.gi.kg.player_location:
            #     assert False, "This code is no longer used"
            #     loc = Location(description=observation)
            #     ev = self.gi.kg.add_location(loc)
            #     self.gi.kg.set_player_location(loc)

        self.step_num += 1

        self.consume_event_stream()

    def select_next_action(self, observation, external_next_action=None):
        if external_next_action:
            next_action = StandaloneAction(external_next_action)
        else:
            if not self.active_module:
                self.elect_new_active_module()
            next_action = self.generate_next_action(observation)
        self._last_action = next_action
        return next_action.text()

    def observe(self, reward, new_obs, done, prev_action=None, idx=None):
        """ Observe could be used for learning from rewards. """
#GVS NOTE 10-25-2020: this is currently not even called for qait , only for ftwc
#        p_valid = self._valid_detector.action_valid(action, new_obs)
#        self.dbg("[VALID] p={:.3f} {}".format(p_valid, clean(new_obs)))

        # NewTransitionEvent unused by current code
        # self.gi.event_stream.push(NewTransitionEvent(prev_obs, prev_action, score, new_obs, terminal))
        if idx is not None:
            assert idx == self._idx
        else:
            idx = self._idx
        if prev_action and self._last_action:
            if prev_action.lower() != self._last_action.text().lower():
                print(f"WARNING: observe() explicit prev_action:'{prev_action}' replacing self._last_action:{self._last_action}")
                self._last_action = StandaloneAction(prev_action)
        if not prev_action and self._last_action:
            prev_action = self._last_action.text()
        if prev_action:
            self.gi.action_recognized(prev_action, new_obs)  # Updates the unrecognized words
        # Print out this step.
        player_location = self.gi.kg.player_location
        _env_name = self.env_name if hasattr(self, 'env_name') else ''
        print(f"**observe: <Step {self.step_num}> [{idx}]{_env_name}  {player_location}: [{prev_action}]   Reward: {reward}")

    def set_ground_truth(self, gt_facts):
        # print("GROUND TRUTH")
        # Reinitialize, build complete GT KG from scratch each time
        self.gi.set_knowledge_graph(KnowledgeGraph(None, groundtruth=True, logger=self.get_logger()), groundtruth=True)
        #TODO (Disabled ground truth)
        # self.gi.gt.update_facts(gt_facts)
        # self.gi.event_stream.push(GroundTruthComplete(groundtruth=True))

    def finalize(self):
        # with open(self.logpath+'.kng', 'w') as f:
        #     f.write(str(self.knowledge_graph)+'\n\n')
        pass

