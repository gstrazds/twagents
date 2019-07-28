from ..valid_detectors.learned_valid_detector import LearnedValidDetector
from ..decision_module import DecisionModule
from ..action import StandaloneAction, PrepareMeal, EatMeal #Eat
from ..event import GroundTruthComplete
from ..game import GameInstance
from .. import gv
from ..util import first_sentence



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
        self.action_sequence = [PrepareMeal, EatMeal]
        # self._action_idx = 0

    def deactivate(self):
        self._active = False
        self._eagerness = 0.0

    def add_required_obj(self, entityname:str):
        print("GTEnder.add_required_obj({})".format(entityname))
        self.required_objs.add(entityname)

    def remove_required_obj(self, entityname:str):
        self.required_objs.discard(entityname)
        self.found_objs.discard(entityname)

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
            if self.have_everything_required(gi.gt):
                print("GT Ender: we possess all required objs:", self.required_objs)
                if self.are_where_we_need_to_be(gi.gt):
                    print("GT Ender: ACTIVATING!")
                    self._active = True
                    self._eagerness = 1.0
            else:
                print("GT Ender: missing some required objs:", self.required_objs, self.found_objs)

    def take_control(self, gi: GameInstance):
        obs = yield
        if not self._active:
            self._eagerness = 0.0
            return None #ends iteration

        if not self.have_everything_required(gi.gt):
            print("[GT ENDER] ABORTING because preconditions are not satisfied", self.required_objs, self.found_objs)
            self.deactivate()
        if not self.are_where_we_need_to_be(gi.gt):
            print("[GT ENDER] ABORTING because location is not correct:", gi.gt.player_location.name)
            self.deactivate()


        response = yield PrepareMeal
        # check to make sure meal object now exists in gi.gt.inventory
        success = gi.gt.inventory.has_entity_with_name('meal')
        self.record(success)
        gv.dbg("[GT ENDER](1.success={}) {} --> {}".format(
            success, PrepareMeal, response))

        response = yield EatMeal
        # check to make sure meal object no longer exists in gi.gt.inventory
        success = gi.gt.inventory.has_entity_with_name()
        self.record(success)
        gv.dbg("[GT ENDER](1.success={}) {} --> {}".format(
            success, EatMeal, response))
