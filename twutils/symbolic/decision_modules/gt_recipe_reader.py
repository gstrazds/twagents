from ..valid_detectors.learned_valid_detector import LearnedValidDetector
from ..decision_module import DecisionModule
from ..action import SingleAction, StandaloneAction, PrepareMeal, Eat, NoOp, Drop
from ..action import Portable
from ..event import GroundTruthComplete, NeedToAcquire, NeedToDo
from ..game import GameInstance
from ..task import SequentialTasks
from ..task_modules import SingleActionTask
from twutils.twlogic import adapt_tw_instr, CUT_WITH, COOK_WITH


def is_portable(objname: str, gi: GameInstance) -> bool:
    entityset = gi.gt.entities_with_name(objname)
    if entityset:
        entity = list(entityset)[0]
        if Portable in entity.attributes:
            return True
        else:
            return False
    else:
        print(f"WARNING: Unable to find entity {objname} in GT knowledge graph")
    return None


def is_already_cut(objname: str, verb: str, gi: GameInstance) -> bool:
    entityset = gi.gt.entities_with_name(objname)
    if entityset:
        entity = list(entityset)[0]
        return entity.state.cuttable and entity.state.is_cut.startswith(verb)
    return False


def is_already_cooked(objname: str, verb: str, gi: GameInstance) -> bool:
    already_cooked = False
    entityset = gi.gt.entities_with_name(objname)
    if entityset:
        entity = list(entityset)[0]
        already_cooked = entity.state.cookable and \
            (entity.state.is_cooked.startswith(verb) or
             verb == 'fry' and entity.state.is_cooked == 'fried')
    return already_cooked



class GTRecipeReader(DecisionModule):
    """
    The Recipe Reader module activates when the player finds the cookbook.
    """
    def __init__(self):
        super().__init__()
        self._eagerness = 0.0
        self.ingredients = []
        self.recipe_steps = []

    def deactivate(self):
        self._eagerness = 0.0

    def clear_all(self):
        self._eagerness = 0.0

    def cookbook_is_here(self, gi: GameInstance):
        cookbook_location = gi.gt.location_of_entity('cookbook')
        return cookbook_location == gi.gt.player_location

    def process_event(self, event, gi: GameInstance):
        """ Process an event from the event stream. """
        if isinstance(event, GroundTruthComplete) and event.is_groundtruth:
            print("GTRecipeReader GroundTruthComplete")
            if not self.recipe_steps and not self.ingredients:  # don't activate a 2nd time
                if self.cookbook_is_here(gi):
                    print("GT Recipe Reader: ACTIVATING!")
                    self._eagerness = 1.0

    def convert_cookbook_step_to_task(self, instr: str, gi: GameInstance):
        instr_words = instr.split()
        enhanced_instr_words, with_objs = adapt_tw_instr(instr_words, gi)
        # TODO: don't chop if already chopped, etc...
        verb = instr.split()[0]
        if verb in CUT_WITH or verb in COOK_WITH:
            obj_words = instr_words[1:]
            if obj_words[0] == 'the':
                obj_words = obj_words[1:]
            obj_name = ' '.join(obj_words)
            with_objs.append(obj_name)
            if verb in CUT_WITH and is_already_cut(obj_name, verb, gi) or \
               verb in COOK_WITH and is_already_cooked(obj_name, verb, gi):
                return SingleActionTask(NoOp, description="REDUNDANT: "+instr, use_groundtruth=True), with_objs  # already cooked or cut
        print("GT RecipeReaderr: mapping <{}> -> {}".format(instr, enhanced_instr_words))

        if instr == 'prepare meal':  # explicit singleton for PrepareMeal
            act = PrepareMeal
        else:
            act = StandaloneAction(" ".join(enhanced_instr_words))

        return SingleActionTask(act, description=instr, use_groundtruth=True), with_objs

    def convert_instructions_to_tasks(self, gi: GameInstance):
        tasks = []
        actions = []
        prep = None
        eat = None
        for instr in self.recipe_steps:
            task, with_objs = self.convert_cookbook_step_to_task(instr, gi)
            assert task is not None
            act = task.action
            if act == NoOp:
                continue    # ignore this one, try to convert next step
            elif act == PrepareMeal:
                prep = SingleActionTask(PrepareMeal, use_groundtruth=True)
                eat = SingleActionTask(StandaloneAction("eat meal"), use_groundtruth=False)
                eat.prereq.required_tasks.append(prep)
            else:
                actions.append((act, instr))
                task = SingleActionTask(act, description=instr, use_groundtruth=True)
                for objname in with_objs:
                    if is_portable(objname, gi):  #TODO: THIS ASSUMES use_groundtruth=True
                        task.prereq.add_required_item(objname)
                    else:
                        task.prereq.add_required_object(objname)
                    tasks.append(task)
        if eat:
            eat.prereq.required_inventory.append('meal')
            for task in tasks:
                prep.prereq.add_required_task(task)
            prep.prereq.add_required_location('kitchen')  # meals can be prepared only in the kitchen\
            for item_name in self.ingredients:
                prep.prereq.add_required_item(item_name)
            return eat
        elif tasks:
            print("WARNING: didn't find Prepare Meal instruction")
            assert prep is not None, "RecipeReader didn't find Prepare Meal instruction"
            return SequentialTasks(tasks, use_groundtruth=True)
        print("ERROR RecipeReader didn't create any Tasks")
        return None

    def take_control(self, gi: GameInstance):
        obs = yield
        if not self._eagerness:
            print(f"WARNING: GTRecipeReader.take_control() with eagerness={self._eagerness}")
            return None #ends iteration

        # cookbook = gi.gt.player_location.get_entity_by_name('cookbook')
        if not self.cookbook_is_here(gi):
            print("[GT RecipeReader] ABORTING because cookbook is not here", gi.gt.player_location)
            self.deactivate()
            return None

        cookbook = gi.gt.get_entity('cookbook')
        response = yield SingleAction('examine', cookbook)
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
                ingredients.append(ingredient)
        if ingredients:
            unneeded_inventory = []  #inventory items that are not listed as ingredients
            already_in_inventory = []  #ingredients that we already have
            for entity in gi.gt.inventory.entities:
                is_in_ingredients = False
                for ingredient in ingredients:
                    if entity.has_name(ingredient):
                        already_in_inventory.append(ingredient)
                        is_in_ingredients = True
                        continue  # check next entity
                if not is_in_ingredients:
                    unneeded_inventory.append(entity)

            self.ingredients = ingredients
            for entity in unneeded_inventory:
                response = yield Drop(entity)

            # if False:  # if Tasks have appropr prereqs, the following should no longer be necessary
            #     for entity_name in already_in_inventory:
            #         ingredients.remove(entity_name)
            #     gi.event_stream.push(NeedToAcquire(objnames=ingredients, groundtruth=True))

        start_of_directions: int = start_of_ingredients + i + 1
        if start_of_directions < len(recipe_lines):
            assert recipe_lines[start_of_directions-1] == 'Directions:'
            for recipe_step in recipe_lines[start_of_directions:]:
                if recipe_step:
                    directions.append(recipe_step)
            if directions:
                self.recipe_steps = directions
                # gi.event_stream.push(NeedSequentialSteps(directions, groundtruth=True))
                main_task = self.convert_instructions_to_tasks(gi)
                if main_task:
                    gi.event_stream.push(NeedToDo(main_task, groundtruth=True))

        self.deactivate()
        return None
