from ..action import Action, SingleAction, StandaloneAction, PrepareMeal, Eat, NoOp, Drop
# from ..event import NeedToDo
# from ..game import GameInstance
from ..knowledge_graph import KnowledgeGraph
from ..task import SequentialTasks, Task
from ..task_modules import SingleActionTask
from ..task_modules.acquire_task import TakeItemTask
from ..task_modules.navigation_task import GoToTask
from twutils.twlogic import adapt_tw_instr, parse_ftwc_recipe, CUT_WITH, COOK_WITH


class RecipeReaderTask(SingleActionTask):
    """
    The Recipe Reader module activates when the player finds the cookbook.
    """
    def __init__(self, use_groundtruth=True):
        super().__init__(act=StandaloneAction("read cookbook"),
            description="RecipeReaderTask", use_groundtruth=False)
        self.ingredients = []
        self.recipe_steps = []
        self.use_groundtruth = use_groundtruth
        self.prereq.add_required_task(GoToTask('cookbook', use_groundtruth=use_groundtruth))
        # self.prereq.required_locations = ['kitchen']

    @property
    def maybe_GT(self):
        return "GT " if self.use_groundtruth else ""

    def cookbook_is_here(self, kg: KnowledgeGraph):
        cookbook_location = kg.location_of_entity_with_name('cookbook')
        return cookbook_location == kg.player_location

    def convert_cookbook_step_to_task(self, instr: str, kg: KnowledgeGraph):
        def check_cooked_state(kgraph):
            retval = kgraph.is_object_cooked(item_name, verb)
            print(f"POSTCONDITION check_cooked_state({item_name}, {verb}) => {retval}")
            return retval

        def check_cut_state(kgraph):
            retval = kgraph.is_object_cut(item_name, verb)
            print(f"POSTCONDITION check_cut_state({item_name}, {verb}) => {retval}")
            return retval

        instr_words = instr.split()
        enhanced_instr_words, with_objs = adapt_tw_instr(instr_words, kg)
        post_checks = []
        item_name = None
        if instr == 'prepare meal':  # explicit singleton for PrepareMeal
            act = PrepareMeal
        else:
            verb = instr.split()[0]
            # TODO: don't chop if already chopped, etc...
            if verb in CUT_WITH or verb in COOK_WITH:
                obj_words = instr_words[1:]
                if obj_words[0] == 'the':
                    obj_words = obj_words[1:]
                item_name = ' '.join(obj_words)
                # with_objs.append(item_name)
            print("GT RecipeReader: mapping <{}> -> {}".format(instr, enhanced_instr_words))
            act = StandaloneAction(" ".join(enhanced_instr_words))
            if verb in CUT_WITH:
                if self.use_groundtruth:
                    if kg.is_object_cut(item_name, verb):
                        act = NoOp
                        instr = "REDUNDANT: " + instr
                else:
                    if kg.get_entity(item_name):  # if we've already encountered this object
                        if check_cut_state(kg):
                            print("REDUNDANT:", instr)
                            # act = NoOp
                            # instr = "REDUNDANT: " + instr
                    # add postconditions to check for already done
                    post_checks.append(check_cut_state)
            elif verb in COOK_WITH:
                if self.use_groundtruth:
                    if kg.is_object_cooked(item_name, verb):
                        act = NoOp
                        instr = "REDUNDANT: " + instr
                else:
                    if kg.get_entity(item_name):  # if we've already encountered this object
                        if check_cooked_state(kg):
                            print("REDUNDANT:", instr)
                    # add postconditions to check for already done
                    post_checks.append(check_cooked_state)

        task = SingleActionTask(act, description=instr, use_groundtruth=self.use_groundtruth)
        if item_name:
            # task.prereq.add_required_item(item_name)
            task.prereq.add_required_task(TakeItemTask(item_name, use_groundtruth=self.use_groundtruth))
        for objname in with_objs:
            if verb in CUT_WITH:   # TODO: this is too specific to TextWorld v1 cooking games
                # or self._knowledge_graph(gi).is_object_portable(objname): # THIS ASSUMES we already know this object (e.g. w/use_groundtruth=True)

                #GVS REFACTORING IN PROGRESS: try replacing required_item with TakeItemTask prereq task
                # task.prereq.add_required_item(objname)
                task.prereq.add_required_task(TakeItemTask(objname, use_groundtruth=self.use_groundtruth))
            else:
                task.prereq.add_required_task(GoToTask(objname, use_groundtruth=self.use_groundtruth))
        if post_checks:
            print("Adding postconditions to task", task, post_checks)
            for func in post_checks:
                task.add_postcondition(func)
        print("convert_cookbook_step_to_task:", task, with_objs, task._postcondition_checks)
        return task, with_objs

    def convert_instructions_to_tasks(self, kg: KnowledgeGraph):
        # kg = self._knowledge_graph(gi)
        tasks = []
        actions = []
        prep = None
        eat = None
        for instr in self.recipe_steps:
            task, with_objs = self.convert_cookbook_step_to_task(instr, kg)
            assert task is not None
            act = task.action
            if act == NoOp:
                continue    # ignore this one, try to convert next step
            elif act == PrepareMeal:
                prep = task  #SingleActionTask(PrepareMeal, use_groundtruth=self.use_groundtruth)
                prep._do_only_once = True
                eat = SingleActionTask(StandaloneAction("eat meal"), use_groundtruth=self.use_groundtruth)
                eat._do_only_once = True
                eat.prereq.required_tasks.append(prep)
            else:
                actions.append((act, instr))
                # task = SingleActionTask(act, description=instr, use_groundtruth=self.use_groundtruth)
                tasks.append(task)
        if eat:
            # eat.prereq.required_inventory.append('meal')  #NOTE: this causes problems because we attempt to search for the meal
            for task in tasks:
                prep.prereq.add_required_task(task)
            prep.prereq.add_required_location('kitchen')  # meals can be prepared only in the kitchen\
            for ingredient_name in self.ingredients:
                # prep.prereq.add_required_item(ingredient_name)
                prep.prereq.add_required_task(TakeItemTask(ingredient_name, use_groundtruth=self.use_groundtruth))

            return eat
        elif tasks:
            print("WARNING: didn't find Prepare Meal instruction")
            assert prep is not None, "RecipeReader didn't find Prepare Meal instruction"
            return SequentialTasks(tasks, use_groundtruth=self.use_groundtruth)
        print("ERROR RecipeReader didn't create any Tasks")
        return None

    def _generate_actions(self, kg) -> Action:
        """ Generates a sequence of actions.
        :type gi: GameInstance
        """
        ignored = yield   # required handshake

        # cookbook = gi.gt.player_location.get_entity_by_name('cookbook')
        if not self.cookbook_is_here(kg):
            print(f"[RecipeReaderTask] ABORTING because cookbook is not here",
                  kg.player_location)
            self.deactivate(kg)
            return None

        cookbook = kg.get_entity('cookbook')
        response = yield self.action  #SingleAction('examine', cookbook)
        # parse the response
        ingredients, directions = parse_ftwc_recipe(response, format='fulltext')
        # recipe_lines = response.split('\n')
        # recipe_lines = list(map(lambda line: line.strip(), recipe_lines))
        # ingredients = []
        # directions = []
        # try:
        #     start_of_ingredients = recipe_lines.index("Ingredients:")
        #     start_of_ingredients += 1
        # except ValueError:
        #     print("RecipeReader failed to find Ingredients in:", response)
        #     self._failed = True
        #     return None
        # for i, ingredient in enumerate(recipe_lines[start_of_ingredients:]):
        #     if ingredient.startswith("Directions"):
        #         break     # end of Ingredients list
        #     if ingredient:
        #         ingredients.append(ingredient)
        # if ingredients:
        #     unneeded_inventory = []  # inventory items that are not listed as ingredients
        #     already_in_inventory = []  #ingredients that we already have
        #     for entity in kg.inventory.entities:
        #         is_in_ingredients = False
        #         for ingredient in ingredients:
        #             if entity.has_name(ingredient):
        #                 already_in_inventory.append(ingredient)
        #                 is_in_ingredients = True
        #                 continue  # check next entity
        #         if not is_in_ingredients:
        #             unneeded_inventory.append(entity)
        #
        #     self.ingredients = ingredients
        #     for entity in unneeded_inventory:
        #         response = yield Drop(entity)

        # start_of_directions: int = start_of_ingredients + i + 1
        # if start_of_directions < len(recipe_lines):
        #     assert recipe_lines[start_of_directions-1] == 'Directions:'
        #     for recipe_step in recipe_lines[start_of_directions:]:
        #         if recipe_step:
        #             directions.append(recipe_step)
        #     if directions:
        #         self.recipe_steps = directions
        #         main_task = self.convert_instructions_to_tasks(kg)
        #         if main_task and self._task_exec:
        #             # kg.event_stream.push(NeedToDo(main_task, groundtruth=self.use_groundtruth))
        #             self._task_exec.queue_task(main_task)

        self.ingredients = ingredients
        self.recipe_steps = directions
        if not self.ingredients and not self.recipe_steps:
            print(f"RECIPE READER FAILED to parse recipe: {response}")
            self._failed = True
            self._done = True
            return None
        if ingredients:
            if self._task_exec:
                drop_unneeded = DropUnneededItemsTask(required_items=ingredients, use_groundtruth=self.use_groundtruth)
                self._task_exec.push_task(drop_unneeded)
            # unneeded_inventory = []  # inventory items that are not listed as ingredients
            # already_in_inventory = []  #ingredients that we already have
            # for entity in kg.inventory.entities:
            #     is_in_ingredients = False
            #     for ingredient in ingredients:
            #         if entity.has_name(ingredient):
            #             already_in_inventory.append(ingredient)
            #             is_in_ingredients = True
            #             continue  # check next entity
            #     if not is_in_ingredients:
            #         unneeded_inventory.append(entity)
            # for entity in unneeded_inventory:
            #     response = yield Drop(entity)
        if directions:
            main_task = self.convert_instructions_to_tasks(kg)
            if main_task and self._task_exec:
                # drop_unneeded = SingleActionTask(StandaloneAction("drop unneeded items"), use_groundtruth=self.use_groundtruth)
                # self._task_exec.push_task(drop_unneeded)
                self._task_exec.queue_task(main_task)
        self.deactivate(kg)
        self._done = True
        return None


class DropUnneededItemsTask(Task):
    """
    The Drop Unneeded items activates after the player reads the cookbook, to free up space in the inventory
    """
    def __init__(self, required_items=None, use_groundtruth=True):
        _items = required_items if required_items else []
        _descr = f"DropUnneededActionsTask{[itemname for itemname in _items]}"
        super().__init__(description=_descr, use_groundtruth=use_groundtruth)
        self.required_items = _items
        self.use_groundtruth = use_groundtruth
        # self.prereq.required_locations = ['kitchen']

    @property
    def maybe_GT(self):
        return "GT " if self.use_groundtruth else ""

    def action_phrase(self) -> str:   # repr similar to a command/action (verb phrase) in the game env
        return "drop unneeded items"

    def check_result(self, result: str, kg: KnowledgeGraph) -> bool:
        return True

    def _generate_actions(self, kg) -> Action:
        """ Generates a sequence of actions.
        :type gi: GameInstance
        """
        ignored = yield   # required handshake


        if self.required_items:
            unneeded_inventory = []  # inventory items that are not listed as ingredients
            already_in_inventory = []  #ingredients that we already have
            for entity in kg.inventory.entities:
                is_in_ingredients = False
                for ingredient in self.required_items:
                    if entity.has_name(ingredient):
                        already_in_inventory.append(ingredient)
                        is_in_ingredients = True
                        continue  # check next entity
                if not is_in_ingredients:
                    unneeded_inventory.append(entity)
            for entity in unneeded_inventory:
                response = yield Drop(entity)
        self.deactivate(kg)
        self._done = True
        return None
