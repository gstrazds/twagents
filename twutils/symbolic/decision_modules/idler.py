from ..valid_detectors.learned_valid_detector import LearnedValidDetector
from ..decision_module import DecisionModule
from ..action import StandaloneAction, SingleAction, DoubleAction
from ..game import GameInstance
from .. import gv
from ..util import first_sentence

standalone_verbs = [
    'get all', 'take all', 'drop all', 'wait', 'yes',
    'look', 'in', 'out', 'climb', 'turn on', 'turn off',
    'use', 'clap', 'get', 'dig', 'swim', 'jump',
    'drink', 'leave', 'put', 'talk', 'hop', 'buy',
    'no', 'dance', 'sleep', 'stand', 'feel', 'sit',
    'pray', 'cross', 'knock', 'open', 'pull', 'push',
    'away', 'kill', 'hide', 'pay', 'type', 'listen',
    'inventory', 'get up'
]

single_object_verbs = [
    'call', 'lock', 'smash', 'kiss', 'free',
    'answer', 'pay', 'make', 'play', 'push',
    'rewind', 'mix', 'sharpen', 'print', 'tap',
    'unlock', 'repair', 'build', 'bribe', 'chew',
    'eat', 'wear', 'think', 'cross', 'cut',
    'slide', 'walk', 'get', 'offer', 'unlight',
    'douse', 'jump', 'buy', 'off', 'remember',
    'shoot', 'oil', 'look', 'operate', 'type',
    'kill', 'clean', 'steal', 'remove', 'turn',
    'press', 'watch', 'wave', 'throw', 'search',
    'exit', 'blow', 'raise', 'cast', 'pluck',
    'unfold', 'open', 'activate', 'ride', 'set',
    'lift', 'arrest', 'pull', 'follow', 'wake',
    'talk', 'hide', 'dial', 'untie', 'start',
    'swing', 'dismount', 'catch', 'feed', 'kick',
    'part', 'inflate', 'touch', 'drink', 'hello',
    'dig', 'rub', 'hit', 'climb', 'swim', 'plug',
    'roll', 'leave', 'put', 'tear', 'break',
    'ring', 'bite', 'warm', 'give', 'say', 'sit',
    'fill', 'shake', 'take', 'enter', 'brandish',
    'light', 'show', 'chop', 'move', 'insert',
    'feel', 'fix', 'burn', 'use', 'stab', 'read',
    'close', 'examine', 'fly', 'hold', 'water',
    'load', 'tie', 'inspect', 'mount', 'empty',
    'connect', 'drop', 'go', 'lower', 'wait',
    'weigh', 'tickle', 'extinguish', 'out', 'on',
    'spray', 'wring', 'pour', 'grab', 'knock on',
    'look under', 'get all from', 'turn on', 'turn off'
]

complex_verbs = [
    ('give','to'), ('tell','to'), ('ask','about'),
    ('put','in'), ('unlock','with'), ('tie','to'),
    ('rub','with'), ('dip','in'), ('ask', 'for'),
    ('kill','with'), ('show','to'), ('chop','with'),
    ('compare','and'), ('throw','at'), ('wet','with'),
    ('get','from'), ('attack','with'), ('dig','with'),
    ('cut','with'), ('insert','in'), ('operate','on'),
    ('open','with'), ('point','at'), ('break','with')
]


class Idler(DecisionModule):
    """
    The Idler module accepts control when no others are willing to.
    """
    def __init__(self, active=False):
        super().__init__()
        self._active = active
        self._valid_detector = LearnedValidDetector()
        self._eagerness = .05


    def process_event(self, event, gi: GameInstance):
        """ Process an event from the event stream. """
        pass


    def get_random_entity(self, gi: GameInstance):
        """ Returns a random entity from the location or inventory. """
        if gi.kg.player_location.entities or gi.kg.inventory.entities:
            return gv.rng.choice(gi.kg.player_location.entities + gi.kg.inventory.entities)
        return None


    def get_standalone_action(self):
        return StandaloneAction(gv.rng.choice(standalone_verbs))


    def get_single_object_action(self, gi: GameInstance):
        entity = self.get_random_entity(gi)
        if not entity:
            return None
        verb = gv.rng.choice(single_object_verbs)
        return SingleAction(verb, entity)


    def get_double_action(self, gi: GameInstance):
        if len(gi.kg.player_location.entities) + len(gi.kg.inventory.entities) <= 1:
            return None
        entity1 = None
        entity2 = None
        count = 0
        while id(entity1) == id(entity2):
            if count == 100:
                return None  # Failsafe
            else:
                count += 1
            entity1 = self.get_random_entity(gi)
            entity2 = self.get_random_entity(gi)
        verb, prep = gv.rng.choice(complex_verbs)
        return DoubleAction(verb, entity1, prep, entity2)

    def get_action(self, gi: GameInstance):
        if not self._active:
            return StandaloneAction('look')
        n = gv.rng.random()
        if n < .1:
            return self.get_standalone_action()
        elif n < .8:
            return self.get_single_object_action(gi)
        else:
            return self.get_double_action(gi)

    def take_control(self, gi: GameInstance):
        obs = yield
        action = self.get_action(gi)
        while action is None or not action.recognized(gi):
            action = self.get_action(gi)
        response = yield action
        p_valid = self._valid_detector.action_valid(action, first_sentence(response))
        ent = None
        if isinstance(action, StandaloneAction):
            gi.action_at_location(action, gi.kg.player_location, p_valid, response)
        else:
            if isinstance(action, SingleAction):
                ent = action.entity
            elif isinstance(action, DoubleAction):
                ent = action.entity1
            else:
                assert False, "Expected either SingleAction or DoubleAction but got {}".format(str(action))
            if ent:
                gi.act_on_entity(action, ent, p_valid, response)
        success = (p_valid > 0.5)
        self.record(success)
        gv.dbg("[IDLER]({}) p={:.2f} {} --> {}".format(
            "val" if success else "inv", p_valid, action, response))
