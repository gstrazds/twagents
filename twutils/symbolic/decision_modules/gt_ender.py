from ..valid_detectors.learned_valid_detector import LearnedValidDetector
from ..decision_module import DecisionModule
from ..action import StandaloneAction, PrepareMeal, Eat, Look #, EatMeal
from ..event import GroundTruthComplete, NeedToAcquire, NeedSequentialSteps, NeedToGoTo
from ..game import GameInstance
# from .. import gv
# from ..util import first_sentence



class GTEnder(DecisionModule):
    """
    The Ender module activates when the player has all the required ingredients.
    """
    def __init__(self, active=False):
        super().__init__()
        self._active = active
        self._eagerness = 0.0
        self.required_objs = set()
        self.found_objs = set()
        self.recipe_steps = []
        # self._action_idx = 0

    def get_eagerness(self, gi: GameInstance):
        """ Returns a float in [0,1] indicating how eager this module is to take
        control. """
        if self.have_everything_required(gi.gt):
            print("GT Ender: we possess all required objs:", self.required_objs)
            if self.are_where_we_need_to_be(gi.gt):
                if not self._active:
                    print("GT Ender: ACTIVATING!")
                self._active = True
                self._eagerness = 1.0
            else:
                if self.required_objs:
                    print("GT Ender: missing some required objs:", self.required_objs, self.found_objs)
        return self._eagerness

    def deactivate(self):
        if self._active:
            print("GT Ender: DEACTIVATING.")
        self._active = False
        self._eagerness = 0.0

    def add_required_obj(self, entityname:str):
        print("GTEnder.add_required_obj({})".format(entityname))
        self.required_objs.add(entityname)

    def remove_required_obj(self, entityname:str):
        self.required_objs.discard(entityname)
        self.found_objs.discard(entityname)

    def add_step(self, acttext:str):
        print("GTEnder.add_step({})".format(acttext))
        if acttext not in self.recipe_steps:
            self.recipe_steps.append(acttext)

    def remove_step(self, acttext:str):
        if acttext in self.recipe_steps:
            self.recipe_steps.remove(acttext)

    def clear_all(self):
        self.required_objs.clear()
        self.found_objs.clear()
        self._eagerness = 0.0
        self._active = False

    def have_everything_required(self, kg):
        if not self.required_objs:
            return False
        for name in self.required_objs:
            e = kg.inventory.get_entity_by_name(name)
            if e:
                self.found_objs.add(name)
        return self.required_objs == self.found_objs

    def are_where_we_need_to_be(self, kg):
        return kg.player_location.name == 'kitchen'

    def process_event(self, event, gi: GameInstance):
        """ Process an event from the event stream. """
        if isinstance(event, GroundTruthComplete) and event.is_groundtruth:
            print("GT complete", event)
            self.get_eagerness(gi)
        elif isinstance(event, NeedToAcquire) and event.is_groundtruth:
            print("GT Need To Acquire:", event.objnames)
            for itemname in event.objnames:
                self.add_required_obj(itemname)
        elif isinstance(event, NeedSequentialSteps) and event.is_groundtruth:
            print("GT Need To Do:", event.steps)
            for acttext in event.steps:
                self.add_step(acttext)

    def check_response(self, response):
        return True

    def convert_instruction_to_action(self, instr: str):
        act = StandaloneAction(instr)
        return act

    def take_control(self, gi: GameInstance):
        obs = yield
        if not self._active:
            self._eagerness = 0.0
            return None #ends iteration

        if not self.have_everything_required(gi.gt):
            print("[GT ENDER] ABORTING because preconditions are not satisfied", self.required_objs, self.found_objs)
            self.deactivate()
            return None

        while self.recipe_steps:
            instr = self.recipe_steps[0]
            step = self.convert_instruction_to_action(instr)
            response = yield step
            success = self.check_response(response)
            if success:
                self.recipe_steps = self.recipe_steps[1:]
            else:
                self.deactivate()
                return None

        if not self.are_where_we_need_to_be(gi.gt):
            print("[GT ENDER] ABORTING because location is not correct:", gi.gt.player_location.name)
            gi.event_stream.push(NeedToGoTo('kitchen', groundtruth=True))
            self.deactivate()
            return None

    # The recipes already contain an instruction step for "prepare meal"
    #     response = yield PrepareMeal
    #     # check to make sure meal object now exists in gi.gt.inventory
        meal = gi.gt.inventory.get_entity_by_name('meal')
        success = meal
    #     self.record(success)
    #     #gv.dbg(
    #     print("[GT ENDER](1.success={}) {} --> {}".format(
    #         success, PrepareMeal, response))
        if not success:
            self.deactivate()
            return None
        EatMeal = Eat(meal)
        response = yield EatMeal
        # check to make sure meal object no longer exists in gi.gt.inventory
        success = not gi.gt.inventory.has_entity_with_name('meal')
        self.record(success)
        #gv.dbg(
        print("[GT ENDER](2.success={}) {} --> {}".format(
            success, EatMeal, response))
