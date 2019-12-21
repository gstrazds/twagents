import os, sys
from ..valid_detectors.learned_valid_detector import LearnedValidDetector
from ..decision_module import DecisionModule
from ..knowledge_graph import *
from ..action import *
from ..gv import rng, dbg
from ..util import tokenize, text_similarity


def _most_similar_loc(description, loc_list):
    """Returns the location from loc_list with that best matches the
    provided description."""
    most_similar = None
    best_similarity = 0
    for loc in loc_list:
        similarity = text_similarity(loc.description, description, substring_match=True)
        if similarity > best_similarity:
            best_similarity = similarity
            most_similar = loc
    return most_similar


def find_most_similar_location(description, kg):
    """ Returns the location with the highest similarity to the given description. """
    possible_name = Location.extract_name(description)
    existing_locs = kg.locations_with_name(possible_name)
    if not existing_locs:
        existing_locs = kg._locations
    return _most_similar_loc(description, existing_locs)



class Navigator(DecisionModule):
    """
    The Navigator is responsible for choosing a navigation action and recording
    the effects of that action.

    Args:
    p_retry: Probability of re-trying failed actions
    eagerness: Default eagerness for this module

    """
    def __init__(self, active=False, p_retry=.3):
        super().__init__()
        self._active = active
        self._nav_actions = [GoNorth, GoSouth, GoWest, GoEast]
            # NorthWest, SouthWest, NorthEast, SouthEast, Up, Down, Enter, Exit]
        self._p_retry = p_retry
        self._valid_detector = LearnedValidDetector()
        self._suggested_directions = []
        self._default_eagerness = 0.1
        self._low_eagerness = 0.01


    def get_mentioned_directions(self, description):
        """ Returns the nav actions mentioned in a description. """
        tokens = tokenize(description)
        return [act for act in self._nav_actions if act.text() in tokens]


    def process_event(self, event, gi: GameInstance):
        """ Process an event from the event stream. """
        pass


    def get_eagerness(self, gi: GameInstance):
        if not self._active:
            return 0.
        if self.get_unexplored_actions(gi.kg.player_location, gi):
            return self._default_eagerness
        return rng.choice([self._low_eagerness, self._default_eagerness])


    def get_unexplored_actions(self, location, gi: GameInstance):
        """ Returns a list of nav actions not yet attempted from a given location. """
        return [act for act in self._nav_actions if act not in location.action_records \
                and act.recognized(gi)]


    def get_successful_nav_actions(self, location, gi: GameInstance):
        """ Returns a list of nav actions that have been successful from the location. """
        return [c.action for c in gi.kg.connections.outgoing(location) if c.action.recognized()]


    def get_failed_nav_actions(self, location, gi: GameInstance):
        """ Returns a list of nav actions that have failed from the location. """
        successful_actions = self.get_successful_nav_actions(location, gi)
        return [act for act in self._nav_actions if act in location.action_records \
                and act not in successful_actions and act.recognized()]


    def get_action(self, gi: GameInstance):
        """
        First try to take an unexplored nav action. If none exist, sample one
        of the successful or failed nav actions.

        """
        loc = gi.kg.player_location

        # If there was a previously suggested direction, try it
        if self._suggested_directions:
            act = rng.choice(self._suggested_directions)
            del self._suggested_directions[:]
            dbg("[NAV] Trying suggested action: {}".format(act))
            return act

        # First try to move in one of the directions mentioned in the description.
        likely_nav_actions = self.get_mentioned_directions(loc.description)
        for act in likely_nav_actions:
            if act not in loc.action_records:
                dbg("[NAV] Trying mentioned action: {}".format(act))
                return act

        # Then try something new
        unexplored = self.get_unexplored_actions(loc, gi)
        if unexplored:
            act = rng.choice(unexplored)
            dbg("[NAV] Trying unexplored action: {}".format(act))
            return act

        # Try a previously successful action
        if rng.random() > self._p_retry:
            successful_actions = self.get_successful_nav_actions(loc, gi)
            if successful_actions:
                act = rng.choice(successful_actions)
                dbg("[NAV] Trying previously successful action: {}".format(act))
                return act

        # Finally, just try something random
        act = rng.choice(self._nav_actions)
        dbg("[NAV] Trying random action: {}".format(act))
        return act

    def relocalize(self, description, gi: GameInstance):
        """Resets the player's location to location best matching the
        provided description, creating a new location if needed. """
        loc = find_most_similar_location(description, gi.kg)
        if loc:
            dbg("[NAV](relocalizing) \"{}\" to {}".format(description, loc))
            gi.kg.set_player_location(loc, gi)
        else:
            dbg("[NAV](relocalizing aborted) \"{}\" to {}".format(description, loc))

    def take_control(self, gi: GameInstance):
        """
        Takes a navigational action and records the resulting transition.

        """
        obs = yield
        curr_loc = gi.kg.player_location
        action = self.get_action(gi)
        response = yield action
        p_valid = self._valid_detector.action_valid(action, response, gi)

        # Check if we've tried this action before
        tried_before = False
        if action in curr_loc.action_records:
            prev_valid, result = curr_loc.action_records[action]
            if result.startswith(response):
                tried_before = True

        gi.kg.action_at_current_location(action, p_valid, response, gi)
        self._suggested_directions = self.get_mentioned_directions(response)
        if self._suggested_directions:
            dbg("[NAV] Suggested Directions: {}".format(self._suggested_directions))
        if action in self._suggested_directions: # Don't try the same nav action again
            self._suggested_directions.remove(action)

        # If an existing locations matches the response, then we're done
        possible_loc_name = Location.extract_name(response)
        existing_locs = gi.kg.locations_with_name(possible_loc_name)
        if existing_locs:
            # If multiple locations match, we need the most similar
            if len(existing_locs) > 1:
                look = yield Look
                existing_loc = _most_similar_loc(look, existing_locs)
            else:
                existing_loc = existing_locs[0]
            dbg("[NAV](revisited-location) {}".format(existing_loc.name))
            gi.kg.add_connection(Connection(curr_loc, action, existing_loc), gi)
            gi.kg.set_player_location(existing_loc, gi)
            return

        # This is either a new location or a failed action
        if tried_before:
            known_destination = gi.kg.connections.navigate(curr_loc, action)
            if known_destination:
                # We didn't reach the expected destination. Likely mislocalized.
                look = yield Look
                self.relocalize(look)
            else: # This has failed previously
                dbg("[NAV-fail] p={:.2f} Response: {}".format(p_valid, response))
        else:
            # This is a new response: do a look to see if we've moved.
            if p_valid < .1:
                dbg("[NAV](Suspected-Invalid) {}".format(response))
                return

            look = yield Look

            p_stay = text_similarity(look, curr_loc.description) / 100.
            p_move = text_similarity(look, response) / 100.
            moved = p_move > p_stay
            dbg("[NAV]({}) p={} {} --> {}".format(
                'val' if moved else 'inv', p_move, action, response))
            self.record(moved)
            if moved:
                # Check if we've moved to an existing location
                possible_loc_name = Location.extract_name(look)
                existing_locs = gi.kg.locations_with_name(possible_loc_name)
                if existing_locs:
                    if len(existing_locs) > 1:
                        existing_loc = _most_similar_loc(look, existing_locs)
                    else:
                        existing_loc = existing_locs[0]
                    dbg("[NAV](revisited-location) {}".format(existing_loc.name))
                    gi.kg.add_connection(Connection(curr_loc, action, existing_loc), gi)
                    gi.kg.set_player_location(existing_loc, gi)
                    return

                # Finally, create a new location
                new_loc = Location(description=look)
                ev = gi.kg.add_location(new_loc)
                gi.kg.add_connection(Connection(curr_loc, action, new_loc), gi)
                gi.kg.set_player_location(new_loc, gi)
                gi.event_stream.push(ev)

