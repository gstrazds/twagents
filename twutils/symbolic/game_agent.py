import os
import logging
import random
from typing import Optional

from .knowledge_graph import KnowledgeGraph
from .task_exec import TaskExec
# from symbolic.entity import Location
from .task_modules import RecipeReaderTask
from .task_modules.navigation_task import ExploreHereTask
from .entity import MEAL
from .action import *
# from twutils.twlogic import DIRECTION_RELATIONS
from twutils.twlogic import get_name2idmap, check_cmd_failed


class TextGameAgent:
    def __init__(self, seed, env_name, idx=0, game=None, output_subdir='.', use_internal_names=False):
        self._idx = idx
        self.setup_logging(env_name, idx, output_subdir)
        self._game = game  # if provided, can do nicer logging
        if game:
            self.map_names2ids = get_name2idmap(game)
        else:
            self.map_names2ids = None
        self.dbg("RandomSeed: {}".format(seed))
        rng = random.Random()  # each game instance gets its own random number generator
        rng.seed(seed)
        observed_knowledge_graph = KnowledgeGraph(groundtruth=False, logger=self.get_logger(), rng=rng,
                                                  use_internal_names=use_internal_names, names2ids=self.map_names2ids)
        groundtruth_graph = None  # KnowledgeGraph(groundtruth=True, logger=self.get_logger(), names2ids=names2ids, rng=rng)
        self.gi = GameInstance(kg=observed_knowledge_graph, gt=groundtruth_graph, rng=rng, logger=self.get_logger())
        # self.knowledge_graph.__init__() # Re-initialize KnowledgeGraph
        # gv.event_stream.clear()
        self.task_exec = None
        self.action_generator = None
        self.first_step = True
        self.step_num = 0
        self._last_action = None
        self.use_internal_names = use_internal_names
        if self.use_internal_names:
            assert self._game is not None
        if env_name:
            self.env_name = env_name
        self._reinitialize_action_generator()
        self.cmd_history = []

    @property
    def last_action(self) -> str:
        return self._last_action.text() if self._last_action else None

    def get_game_data(self):
        return self._game

    def reset(self, forget_everything=False):
        names2ids = get_name2idmap(self._game) if self._game else None

        # groundtruth_graph = KnowledgeGraph(None, groundtruth=True, names2ids=names2ids, logger=self.get_logger())    # start fresh with empty graph
        groundtruth_graph = None
        if forget_everything:
            observed_knowledge_graph = KnowledgeGraph(groundtruth=False, names2ids=names2ids, logger=self.get_logger())    # start fresh with empty graph
        else:
            observed_knowledge_graph = self.gi.kg   # reuse existing knowledge
            observed_knowledge_graph.reset()        # bet restore to initial state (start of episode)
        self.gi = GameInstance(kg=observed_knowledge_graph, gt=groundtruth_graph, logger=self.get_logger())
        self._reinitialize_action_generator()
        self.first_step = True
        self.step_num = 0
        self.cmd_history = []

    def set_objective(self, objective : str):
        if objective == 'eat meal':
            assert self.step_num == 0
            # FIXME: initialization HACK for MEAL
            _gi = self.gi
            if not _gi.kg.get_entity('meal'):
                meal = _gi.kg.create_new_object('meal', MEAL)
                _gi.kg._nowhere.add_entity(meal)  # the meal doesn't yet exist in the world
            if self.task_exec:
                _use_groundtruth = False
                task_list = [ExploreHereTask(use_groundtruth=_use_groundtruth),
                             RecipeReaderTask(use_groundtruth=_use_groundtruth)]
                for task in task_list:
                    self.task_exec.queue_task(task)
        else:
            assert False, f"UNIMPLEMENTED objective: {objective}"

    def _reinitialize_action_generator(self):
        self.task_exec = TaskExec(True)
        self.action_generator = None
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
        #self.warn("TEST WARN LOGGING: THIS SHOULD GET LOGGED!")

    def get_logger(self):
        return logging.getLogger(self._logger_name)

    def dbg(self, msg, *args, **kwargs):
        self.get_logger().debug(msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        print(msg, *args)
        self.get_logger().info(msg, *args, **kwargs)

    def warn(self, msg, *args, **kwargs):
        print("WARNING!", msg, *args)
        self.get_logger().warning(msg, *args, **kwargs)

    def restart_task_exec(self):
        """ Reinitializes the stack/queue of tasks. """
        print("restart_task_exec")
        self.action_generator = self.task_exec.take_control(self.gi)
        self.action_generator.send(None)  # handshake with initial argless yield

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
                self.warn(msg)
                # return "Look"
                next_action = Look
            else:
                try:
                    next_action = self.action_generator.send(observation)
                    if next_action:
                        msg = f"[game_agent] Step:{self.step_num} (generate_next_action): -> |{next_action}|"
                        self.dbg(msg)
                except StopIteration:
                    # self.consume_event_stream()
                    self.restart_task_exec()
        return next_action

    # def choose_next_action(self, observation, observable_facts=None, prev_action=None):
    #     print("DEPRECATION WARNING: game_agent.choose_next_action()")
    #     assert False
    #     self.update_kg(observation, observable_facts=observable_facts, prev_action=prev_action)
    #     return self.select_next_action(observation, external_next_action=None)

    def preprocess_observation(self, obstxt:str) -> str:
        if obstxt:
            obstxt.strip()
            if "Your score has just" in obstxt:
                obstxt = '\n'.join(
                    [line for line in obstxt.split('\n') if not line.startswith("Your score has just")]
                ).strip()
        else:
            obstxt = ''
        return obstxt

    def update_kg(self, observation, observable_facts=None, prev_action=None):

        if prev_action and self._last_action:
            if prev_action.lower() != self._last_action.text().lower() and \
                    not self._last_action.text().startswith("answer:"):
                self.warn(f"WARNING: prev_action:|{prev_action}| != self._last_action: |{self._last_action.text()}|")
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

        obstxt = self.preprocess_observation(observation)
        if observable_facts and self.gi.kg:
            obstxt = self.gi.kg.update_facts(obstxt, observable_facts, prev_action=prev_action)
        self.step_num += 1
        # self.consume_event_stream()
        return obstxt

    def select_next_action(self, observation, external_next_action=None):
        if external_next_action:
            next_action = StandaloneAction(external_next_action)
        else:
            if not self.action_generator:
                self.restart_task_exec()
            next_action = self.generate_next_action(observation)
        self._last_action = next_action
        cmdstr = next_action.text()
        if self.gi.kg:
            cmdstr = self.gi.kg.cannonicalize_command(cmdstr)
        return cmdstr

    def observe(self, reward: float, new_obs, done: bool, prev_action: Optional[str] = None,  idx=None):
# GVS NOTE 10-25-2020: this is currently not even called for qait , only for ftwc
#        p_valid = self._valid_detector.action_valid(action, new_obs)
#        self.dbg("[VALID] p={:.3f} {}".format(p_valid, clean(new_obs)))

        if idx is not None:
            assert idx == self._idx
        else:
            idx = self._idx
        if prev_action and self._last_action:
            if prev_action.lower() != self._last_action.text().lower():
                warnmsg = f"WARNING: observe() explicit prev_action:'{prev_action}' replacing self._last_action:{self._last_action}"
                print(warnmsg)
                if not prev_action.startswith(self._last_action.text().lower()):
                    assert False, warnmsg
                self._last_action = StandaloneAction(prev_action)
        if not prev_action and self._last_action:
            prev_action = self._last_action.text()
        if prev_action:
            self.gi.action_recognized(prev_action, new_obs)  # Updates the unrecognized words
        # Print out this step.
        player_location = self.gi.kg.player_location
        _env_name = self.env_name if hasattr(self, 'env_name') else ''
        cmd_ok = not check_cmd_failed(prev_action, new_obs, reward)
        self.dbg(f"**observe: <Step {self.step_num}> [{idx}]{_env_name}  {player_location}: [{prev_action}] {'OK' if cmd_ok else 'FAILED'} Reward: {reward}")
        self.cmd_history.append((prev_action, str(player_location), cmd_ok, reward))

    def set_ground_truth(self, gt_facts):
        # print("GROUND TRUTH")
        # Reinitialize, build complete GT KG from scratch each time
        # names2ids = get_name2idmap(self._game) if self._game else None
        # self.gi.set_knowledge_graph(KnowledgeGraph(None, groundtruth=True, logger=self.get_logger()), groundtruth=True, names2ids=names2ids)
        #TODO/DONE (Disabled ground truth)
        # self.gi.gt.update_facts(None, gt_facts)
        # self.gi.event_stream.push(GroundTruthComplete(groundtruth=True))

        print("IGNORING set_ground_truth(gt_facts)")
        pass

    def finalize(self):
        # with open(self.logpath+'.kng', 'w') as f:
        #     f.write(str(self.knowledge_graph)+'\n\n')
        pass

