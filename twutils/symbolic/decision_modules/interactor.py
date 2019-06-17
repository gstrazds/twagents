from ..valid_detectors.learned_valid_detector import LearnedValidDetector
from ..affordance_extractors.tw_affordance_extractor import TWAffordanceExtractor
from ..decision_module import DecisionModule
from ..gv import GameInstance, dbg
from ..util import clean, first_sentence
from ..action import SingleAction, DoubleAction


class Interactor(DecisionModule):
    """
    The Interactor creates actions designed to interact with objects
    at the current location.
    """
    def __init__(self, active=False):
        super().__init__()
        self._active = active
        self._valid_detector = LearnedValidDetector()
        self._affordance_extractor = TWAffordanceExtractor()
        self.best_action = None
        self._eagerness = 0.
        self.actions_that_caused_death = {}

    def process_event(self, event, gi: GameInstance):
        pass

    def get_eagerness(self, gi: GameInstance):
        if not self._active:
            return 0.
        self.best_action = None
        self._eagerness = 0.
        max_eagerness = 0.

        # Consider single-object actions.
        for entity in gi.kg.player_location.entities + gi.kg.inventory.entities:
            for action, prob in self._affordance_extractor.extract_single_object_actions(entity):
                if prob <= max_eagerness:
                    break
                if entity.has_action_record(action) or \
                        (not action.recognized(gi)) or \
                        (action in self.actions_that_caused_death) or \
                        ((action.verb == 'take') and (entity in gi.kg.inventory.entities)):  # Need to promote to Take.
                    continue
                max_eagerness = prob
                self.best_action = action
                break

        # Consider double-object actions.
        for entity1 in gi.kg.player_location.entities + gi.kg.inventory.entities:
            for entity2 in gi.kg.player_location.entities + gi.kg.inventory.entities:
                if entity1 != entity2:
                    for action, prob in self._affordance_extractor.extract_double_object_actions(entity1, entity2):
                        if prob <= max_eagerness:
                            break
                        if entity1.has_action_record(action) or \
                                (not action.recognized(gi)) or \
                                (action in self.actions_that_caused_death):
                            continue
                        max_eagerness = prob
                        self.best_action = action
                        break

        self._eagerness = max_eagerness
        return self._eagerness

    def take_control(self, gi: GameInstance):
        obs = yield  # obs = previous observation -- seems to be ignored here

        # Failsafe checks
        if self._eagerness == 0.:  # Should never happen anyway.
            self.get_eagerness(gi)  # But if it does, try finding a best action.
            if self._eagerness == 0.:
                return  # If no good action can be found, simply return without yielding.

        action = self.best_action
        self.best_action = None
        self._eagerness = 0.

        response = yield action
        p_valid = action.validate(response)
        ent = None
        if p_valid is None:
            p_valid = self._valid_detector.action_valid(action, first_sentence(response))
        if isinstance(action, SingleAction):
            ent = action.entity
        elif isinstance(action, DoubleAction):
            ent = action.entity1
        if ent:
            gi.act_on_entity(action, ent, p_valid, response)
        else:
            print("WARNING Interactor.take_control(): expecting SingleAction or DoubleAction, but got:", action)
        success = (p_valid > 0.5)
        self.record(success)
        if success:
            action.apply(gi)
        dbg("[INT]({}) p={:.2f} {} --> {}".format(
            "val" if success else "inv", p_valid, action, response))

        if ('RESTART' in response and 'RESTORE' in response and 'QUIT' in response) or ('You have died' in response):
            if action not in self.actions_that_caused_death:
                self.actions_that_caused_death[action] = True  # Remember actions that cause death.
