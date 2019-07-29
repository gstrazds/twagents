from ..valid_detectors.learned_valid_detector import LearnedValidDetector
from ..decision_module import DecisionModule
from ..action import SingleAction
from ..event import GroundTruthComplete, NeedToAcquire
from ..game import GameInstance
from .. import gv
from ..util import first_sentence



class GTRecipeReader(DecisionModule):
    """
    The Recipe Reader module activates when the player finds the cookbook.
    """
    def __init__(self, active=False):
        super().__init__()
        self._active = active
        self._eagerness = 0.0

    def deactivate(self):
        self._active = False
        self._eagerness = 0.0

    def clear_all(self):
        self._active = False
        self._eagerness = 0.0

    def process_event(self, event, gi: GameInstance):
        """ Process an event from the event stream. """
        if isinstance(event, GroundTruthComplete) and event.is_groundtruth:
            print("GTRecipeReader GroundTruthComplete")
            player_loc = gi.gt.player_location
            cookbook = gi.gt.player_location.get_entity_by_name('cookbook')
            if cookbook:
                print("GT Recipe Reader: ACTIVATING!")
                self._active = True
                self._eagerness = 1.0
        # elif isinstance(event, NeedToAcquire) and event.is_groundtruth:
        #     print("GTRecipeReader Need To Acquire", event.objnames)
        #     for itemname in event.objnames:
        #         self.add_required_obj(itemname)

    def take_control(self, gi: GameInstance):
        obs = yield
        if not self._active:
            self._eagerness = 0.0
            return None #ends iteration

        cookbook = gi.gt.player_location.get_entity_by_name('cookbook')
        if not cookbook:
            print("[GT RecipeReader] ABORTING because cookbook is not here", gi.gt.player_location)
            self.deactivate()

        response = yield SingleAction('read', cookbook)
        # parse the response
        recipe_lines = response.split('\n')
        recipe_lines = map(lambda line: line.strip(), recipe_lines)
        ingredients = []
        try:
            start_of_ingredients = recipe_lines.index("Ingredients:")
            start_of_ingredients += 1
        except ValueError:
            print("GT RecipeReader failed to find Ingredients in:", response)
            return None
        for ingredient in recipe_lines[start_of_ingredients:]:
            if ingredient or ingredient.startswith("Directions"):
                break     # end of Ingredients list
            ingredients.append(ingredient)
        if ingredients:
            gi.event_stream.push(NeedToAcquire(objnames=ingredients))
            self.deactivate()