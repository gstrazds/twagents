from ..valid_detectors.learned_valid_detector import LearnedValidDetector
from ..decision_module import DecisionModule
from ..action import SingleAction, Drop
from ..event import GroundTruthComplete, NeedToAcquire, NeedSequentialSteps
from ..game import GameInstance

class GTRecipeReader(DecisionModule):
    """
    The Recipe Reader module activates when the player finds the cookbook.
    """
    def __init__(self, active=False):
        super().__init__()
        self._active = active
        self._eagerness = 0.0
        self.ingredients = []
        self.recipe_steps = []

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
            if not self.recipe_steps and not self.ingredients:  # don't activate a 2nd time
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
            return None

        response = yield SingleAction('read', cookbook)
        # parse the response
        recipe_lines = response.split('\n')
        recipe_lines = list(map(lambda line: line.strip(), recipe_lines))
        ingredients = []
        directions = []
        try:
            start_of_ingredients = recipe_lines.index("Ingredients:")
            start_of_ingredients += 1
        except ValueError:
            print("GT RecipeReader failed to find Ingredients in:", response)
            return None
        for i, ingredient in enumerate(recipe_lines[start_of_ingredients:]):
            if ingredient.startswith("Directions"):
                break     # end of Ingredients list
            if ingredient:
                already_in_inventory = False
                for entity in gi.gt.inventory.entities:
                    if not entity.has_name(ingredient):
                        response = yield Drop(entity)
                    else:
                        already_in_inventory = True
                if not already_in_inventory:
                    ingredients.append(ingredient)
        if ingredients:
            self.ingredients = ingredients
            gi.event_stream.push(NeedToAcquire(objnames=ingredients, groundtruth=True))
        start_of_directions = start_of_ingredients + i + 1
        if start_of_directions < len(recipe_lines):
            assert recipe_lines[start_of_directions-1] == 'Directions:'
            for recipe_step in recipe_lines[start_of_directions:]:
                if recipe_step:
                    directions.append(recipe_step)

            if directions:
                self.recipe_steps = directions
                gi.event_stream.push(NeedSequentialSteps(directions, groundtruth=True))
        self.deactivate()
        return None
