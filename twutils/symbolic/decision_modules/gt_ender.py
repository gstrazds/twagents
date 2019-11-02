from ..valid_detectors.learned_valid_detector import LearnedValidDetector
from ..decision_module import DecisionModule
from ..action import StandaloneAction, PrepareMeal, Eat, Look #, EatMeal
from ..action import Portable
from ..event import GroundTruthComplete, NeedToAcquire, NoLongerNeed
from ..event import NeedSequentialSteps, NeedToGoTo, NeedToFind  #, AlreadyDone
from ..task_modules import SingleActionTask
from ..game import GameInstance
# from ..util import first_sentence

COOK_WITH = {
    "grill": "BBQ",
    "bake": "oven",
    "roast": "oven",
    "fry": "stove",
    "toast": "toaster",
}

CUT_WITH = {
    "chop": "knife",
    "dice": "knife",
    "slice": "knife",
    "mince": "knife",
}


def convert_cooking_instruction(words, device: str, change_verb=None):
    words_out = words.copy()
    words_out.append("with")
    words_out.append(device)
    if change_verb:
        words_out[0] = change_verb  # convert the verb to generic "cook" (the specific verbs don't work as is in TextWorld)
    return words_out


def adapt_tw_instr(words: str, gi) -> str:
    # if instr.startswith("chop ") or instr.startswith("dice ") or instr.startswith("slice "):
    #     return instr + " with the knife", ["knife"]
    # words = instr.split()
    with_objs = []
    if words[0] in COOK_WITH:
        device = COOK_WITH[words[0]]
        with_objs.append(device)
        return convert_cooking_instruction(words, device, change_verb="cook"), with_objs
    elif words[0] in CUT_WITH:
        device = CUT_WITH[words[0]]
        with_objs.append(device)
        return convert_cooking_instruction(words, device), with_objs
    else:
        return words, []



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
                self._eagerness = 0.95
            else:
                print("GT Ender: not at correct location")
                if self.required_objs:
                    print("GT Ender -- required:", self.required_objs, "found:", self.found_objs)
                gi.event_stream.push(NeedToGoTo('kitchen', groundtruth=True))
                self.deactivate()
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
        print("GTEnder.remove_required_obj({})".format(entityname))
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
        if not self.required_objs and not self.recipe_steps:
            return False
        found = set()
        for name in self.required_objs:
            e2 = kg.player_location.get_entity_by_name(name)
            e = kg.inventory.get_entity_by_name(name)
            if e or (e2 and Portable not in e2.attributes):
                # if e2:
                #     attrs = [ attr.name for attr in e2.attributes ]
                #     print(e2, '----------- attrs:', attrs)
                found.add(name)
        self.found_objs = found
        return self.required_objs == self.found_objs

    def are_where_we_need_to_be(self, kg):
        if not self.recipe_steps:
            return True
        if self.recipe_steps[0].startswith('prepare'):
            return kg.player_location.name == 'kitchen'
        return True

    def process_event(self, event, gi: GameInstance):
        """ Process an event from the event stream. """
        # if isinstance(event, GroundTruthComplete) and event.is_groundtruth:
        #     print("GT complete", event)
        #     self.get_eagerness(gi)
        if isinstance(event, NeedToAcquire) and event.is_groundtruth:
            print("GT Required Objects:", event.objnames)
            for itemname in event.objnames:
                # gi.gt.entities_with_name()
                self.add_required_obj(itemname)
        elif isinstance(event, NeedToFind) and event.is_groundtruth:
            print("GT Need to Find Objects:", event.objnames)
            for itemname in event.objnames:
                # gi.gt.entities_with_name()
                self.add_required_obj(itemname)
        elif isinstance(event, NoLongerNeed) and event.is_groundtruth:
            print("GT Not Needed Objects:", event.objnames)
            for itemname in event.objnames:
                self.remove_required_obj(itemname)
        elif isinstance(event, NeedSequentialSteps) and event.is_groundtruth:
            print("GT Need To Do:", event.steps)
            for acttext in event.steps:
                self.add_step(acttext)
        # elif isinstance(event, AlreadyDone) and event.is_groundtruth:
        #     print("GT AlreadyDone:", event.instr_step)
        #     self.remove_step(event.instr_step)

    def check_response(self, response):
        return True

    def convert_next_instruction_to_action(self, gi: GameInstance):
        while self.recipe_steps:
            instr = self.recipe_steps[0]
            instr_words = instr.split()
            enhanced_instr_words, with_objs = adapt_tw_instr(instr_words, gi)
            # TODO: don't chop if already chopped, etc...
            verb = instr.split()[0]
            if verb in CUT_WITH or verb in COOK_WITH:
                obj_words = instr_words[1:]
                if obj_words[0] == 'the':
                    obj_words = obj_words[1:]
                obj_name = ' '.join(obj_words)
                entity = gi.gt.inventory.get_entity_by_name(obj_name)
                if not entity:
                    print(f"WARNING: expected but failed to find {obj_name} in Inventory!")
                    gi.event_stream.push(NeedToAcquire(objnames=[obj_name], groundtruth=True))
                    return None, None  # maybe we can re-acquire it
                    # continue  #try the next instruction
                if verb in CUT_WITH and entity.state.cuttable and entity.state.is_cut.startswith(verb) \
                 or verb in COOK_WITH and entity.state.cookable and entity.state.is_cooked.startswith(verb)\
                 or verb == 'fry' and entity.state.cookable and entity.state.is_cooked == 'fried':
                    # # gi.event_stream.push(AlreadyDone(instr, groundtruth=True))
                    self.remove_step(instr)
                    if with_objs:
                        for obj in with_objs:
                            self.remove_required_obj(obj)
                    continue  #already cooked
            print("GT Ender: mapping <{}> -> {}".format(instr, enhanced_instr_words))
            if with_objs:
                # TODO: if object is not takeable, navigate to object
                for objname in with_objs:
                    entityset = gi.gt.entities_with_name(objname)
                    if entityset:
                        entity = list(entityset)[0]
                        if Portable in entity.attributes:
                            if not gi.gt.inventory.get_entity_by_name(objname):
                                gi.event_stream.push(NeedToAcquire(objnames=[objname], groundtruth=True))
                                return None, None
                        else:  # need to navigate to this object
                            if not gi.gt.player_location.get_entity_by_name(objname):
                                gi.event_stream.push(NeedToFind(objnames=[objname], groundtruth=True))
                                return None, None
            act = StandaloneAction(" ".join(enhanced_instr_words))
            self.remove_step(instr)
            if with_objs:
                for obj in with_objs:
                    self.remove_required_obj(obj)
                gi.event_stream.push(NoLongerNeed(objnames=with_objs, groundtruth=True))
            return act, instr
        return None, None

    def take_control(self, gi: GameInstance):
        obs = yield
        if not self._active:
            self._eagerness = 0.0
            return None #ends iteration

        while self.recipe_steps:
            if not self.have_everything_required(gi.gt):
                print("[GT ENDER] ABORTING because preconditions are not satisfied", self.required_objs,
                      self.found_objs)
                self.deactivate()
                return None
            if not self.are_where_we_need_to_be(gi.gt):
                print("[GT ENDER] ABORTING because location is not correct:", gi.gt.player_location.name)
                gi.event_stream.push(NeedToGoTo('kitchen', groundtruth=True))
                self.deactivate()
                return None
            step, instr = self.convert_next_instruction_to_action(gi)
            if step:
                task = SingleActionTask(step, description=instr)
                response = yield step
                success = self.check_response(response)
            else:
                print("GT Ender FAILED to convert next step to action", self.recipe_steps)
                success = False
            if step and success:
                # self.recipe_steps = self.recipe_steps[1:]
                print("GT Ender recipe_steps <= ", self.recipe_steps)
            else:
                print("GT Ender stopping: failed to execute next step")
                self.deactivate()
                return None

    # The recipes already contain an instruction step for "prepare meal"
    #     response = yield PrepareMeal
    #     # check to make sure meal object now exists in gi.gt.inventory
        meal = gi.gt.inventory.get_entity_by_name('meal')
        success = meal is not None
    #     self.record(success)
    #     #gv.dbg(
    #     print("[GT ENDER](1.success={}) {} --> {}".format(
    #         success, PrepareMeal, response))
        if not success:
            print("GTEnder FAILED TO PREPARE MEAL")
            gi.event_stream.push(NeedToGoTo('kitchen', groundtruth=True))
            if 'prepare meal' not in self.recipe_steps:
                self.recipe_steps.append('prepare meal')
            self.deactivate()
            return None
        act = Eat(meal)
        response = yield act
        # check to make sure meal object no longer exists in gi.gt.inventory
        success = not gi.gt.inventory.has_entity_with_name('meal')
        self.record(success)
        #gv.dbg(
        print("[GT ENDER](2.success={}) {} --> {}".format(
            success, act, response))
