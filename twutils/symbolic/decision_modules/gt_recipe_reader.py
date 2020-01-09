from ..valid_detectors.learned_valid_detector import LearnedValidDetector
from ..decision_module import DecisionModule
from ..action import SingleAction, StandaloneAction, PrepareMeal, Eat, NoOp, Drop
from ..action import Portable
from ..event import GroundTruthComplete, NeedToDo
from ..game import GameInstance
from ..task import SequentialTasks
from ..task_modules import SingleActionTask
from twutils.twlogic import adapt_tw_instr, CUT_WITH, COOK_WITH


class GTRecipeReader(DecisionModule):
    """
    The Recipe Reader module activates when the player finds the cookbook.
    """
    def __init__(self, use_groundtruth=True):
        super().__init__()
        self._eagerness = 0.0
        self.ingredients = []
        self.recipe_steps = []
        self.use_groundtruth = use_groundtruth

    @property
    def maybe_GT(self):
        return "GT " if self.use_groundtruth else ""

    def _knowledge_graph(self, gi):
        if self.use_groundtruth:
            return gi.gt
        return gi.kg

    def deactivate(self):
        self._eagerness = 0.0

    def clear_all(self):
        self._eagerness = 0.0

    def cookbook_is_here(self, gi: GameInstance):
        cookbook_location = self._knowledge_graph(gi).location_of_entity_with_name('cookbook')
        return cookbook_location == self._knowledge_graph(gi).player_location

    def process_event(self, event, gi: GameInstance):
        """ Process an event from the event stream. """
        if isinstance(event, GroundTruthComplete) and event.is_groundtruth:
            print("GTRecipeReader GroundTruthComplete")
            if not self.recipe_steps and not self.ingredients:  # don't activate a 2nd time
                if self.cookbook_is_here(gi):
                    print("GT Recipe Reader: ACTIVATING!")
                    self._eagerness = 1.0

    def convert_cookbook_step_to_task(self, instr: str, gi: GameInstance):
        def check_cooked_state(kgraph):
            retval = kgraph.is_object_cooked(obj_name, verb)
            print(f"POSTCONDITION check_cooked_state({obj_name}, {verb}) => {retval}")
            return retval

        def check_cut_state(kgraph):
            retval = kgraph.is_object_cut(obj_name, verb)
            print(f"POSTCONDITION check_cut_state({obj_name}, {verb}) => {retval}")
            return retval

        instr_words = instr.split()
        enhanced_instr_words, with_objs = adapt_tw_instr(instr_words, gi)
        post_checks = []

        if instr == 'prepare meal':  # explicit singleton for PrepareMeal
            act = PrepareMeal
        else:
            verb = instr.split()[0]
            # TODO: don't chop if already chopped, etc...
            if verb in CUT_WITH or verb in COOK_WITH:
                obj_words = instr_words[1:]
                if obj_words[0] == 'the':
                    obj_words = obj_words[1:]
                obj_name = ' '.join(obj_words)
                with_objs.append(obj_name)
            print("GT RecipeReader: mapping <{}> -> {}".format(instr, enhanced_instr_words))
            act = StandaloneAction(" ".join(enhanced_instr_words))
            if verb in CUT_WITH:
                if self.use_groundtruth:
                    if gi.gt.is_object_cut(obj_name, verb):
                        act = NoOp
                        instr = "REDUNDANT: " + instr
                elif gi.kg.get_entity(obj_name):  # if we've already encountered this object
                    if check_cut_state(gi.kg):
                        act = NoOp
                        instr = "REDUNDANT: " + instr
                else:
                    # add postconditions to check for already done
                    post_checks.append(check_cut_state)
            elif verb in COOK_WITH:
                if self.use_groundtruth:
                    if gi.gt.is_object_cooked(obj_name, verb):
                        act = NoOp
                        instr = "REDUNDANT: " + instr
                elif gi.kg.get_entity(obj_name):  # if we've already encountered this object
                    if check_cooked_state(gi.kg):
                        act = NoOp
                        instr = "REDUNDANT: " + instr
                else:
                    # add postconditions to check for already done
                    post_checks.append(check_cooked_state)

        task = SingleActionTask(act, description=instr, use_groundtruth=self.use_groundtruth)
        if post_checks:
            print("Adding postconditions to task", task, post_checks)
            for func in post_checks:
                task.add_postcondition(func)
        print(task, task._postcondition_checks)
        return task, with_objs

    def convert_instructions_to_tasks(self, gi: GameInstance):
        kg = self._knowledge_graph(gi)
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
                prep = task  #SingleActionTask(PrepareMeal, use_groundtruth=self.use_groundtruth)
                eat = SingleActionTask(StandaloneAction("eat meal"), use_groundtruth=self.use_groundtruth)
                eat.prereq.required_tasks.append(prep)
            else:
                actions.append((act, instr))
                # task = SingleActionTask(act, description=instr, use_groundtruth=self.use_groundtruth)
                for objname in with_objs:
                    if kg.is_object_portable(objname):  #TODO: THIS ASSUMES we already know this object (e.g. w/use_groundtruth=True)
                        task.prereq.add_required_item(objname)
                    else:
                        task.prereq.add_required_object(objname)
                    tasks.append(task)
        if eat:
            # eat.prereq.required_inventory.append('meal')  #NOTE: this causes problems because we attempt to search for the meal
            for task in tasks:
                prep.prereq.add_required_task(task)
            prep.prereq.add_required_location('kitchen')  # meals can be prepared only in the kitchen\
            for item_name in self.ingredients:
                prep.prereq.add_required_item(item_name)
            return eat
        elif tasks:
            print("WARNING: didn't find Prepare Meal instruction")
            assert prep is not None, "RecipeReader didn't find Prepare Meal instruction"
            return SequentialTasks(tasks, use_groundtruth=self.use_groundtruth)
        print("ERROR RecipeReader didn't create any Tasks")
        return None

    def take_control(self, gi: GameInstance):
        obs = yield
        if not self._eagerness:
            print(f"WARNING: GTRecipeReader.take_control() with eagerness={self._eagerness}")
            return None #ends iteration

        # cookbook = gi.gt.player_location.get_entity_by_name('cookbook')
        if not self.cookbook_is_here(gi):
            print(f"[GT RecipeReader] ABORTING because cookbook is not here",
                  self._knowledge_graph(gi).player_location)
            self.deactivate()
            return None

        cookbook = self._knowledge_graph(gi).get_entity('cookbook')
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
                main_task = self.convert_instructions_to_tasks(gi)
                if main_task:
                    gi.event_stream.push(NeedToDo(main_task, groundtruth=self.use_groundtruth))

        self.deactivate()
        return None
