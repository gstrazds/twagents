from symbolic import gv
from symbolic import event
# from symbolic import action
# from symbolic import knowledge_graph
from symbolic import util


def get_unrecognized(action, response):
    """
    Returns an unrecognized word based on the response or empty string.

    Args:
      action: The action that was taken
      response: The textual response from the game

    Returns: string containing the unrecognized word or
    empty string if recognized.

    """
    # if isinstance(action, Action):
    if hasattr(action, "text"):
        action = action.text()
    for p in util.COMPILED_UNRECOGNIZED_REGEXPS:
        match = p.match(response)
        if match:
            if match.groups():
                return match.group(1)
            else:
                return action.split(' ')[0]
    return ''


class GameInstance:
    def __init__(self, kg):  # kg : knowledge_graph.KnowledgeGraph
        self.event_stream = event.EventStream()
        self._unrecognized_words = gv.ILLEGAL_ACTIONS[:]
        #
        self.kg = kg

    def action_at_location(self, action, location, p_valid, result_text):
        ev = location.add_action_record(action, p_valid, result_text)
        self.event_stream.push(ev)

    def entity_at_location(self, entity, location):
        ev = location.add_entity(entity)
        if ev:
            self.event_stream.push(ev)

    def entity_at_entity(self, entity1, entity2):
        ev = entity1.add_entity(entity2)
        if ev:
            self.event_stream.push(ev)

    def act_on_entity(self, action, entity, p_valid, result_text):
        ev = entity.add_action_record(action, p_valid, result_text)
        if ev:
            self.event_stream.push(ev)

    def add_entity_attribute(self, entity, attribute):
        ev = entity.add_attribute(attribute)
        if ev:
            self.event_stream.push(ev)

    def move_entity(self, entity, origin, dest):
        """ Moves entity from origin to destination. """
        assert origin.has_entity(entity), \
            "Can't move entity {} that isn't present at origin {}" \
            .format(entity, origin)
        origin.del_entity(entity)
        ev0 = dest.add_entity(entity)
        if ev0 is not None:
            msg = "unexpected NewEntityEvent from dest.add_entity() dest={} event={}".format(dest, ev0)
            print("!!!WARNING: "+msg)
            assert False, msg
        self.event_stream.push(event.EntityMovedEvent(entity, origin, dest))

    def includes_unrecognized_words(self, textline):
        for word in textline.split(' '):
            if word in self._unrecognized_words:
                return False

    def action_recognized(self, action, response):
        """
        Returns True if the action was recognized based on the response.
        Returns False if the action is not recognized and appends it to
        the list of unrecognized_words.

        """
        unrecognized_word = get_unrecognized(action, response)
        if unrecognized_word:
            if unrecognized_word not in self._unrecognized_words:
                gv.dbg("[UTIL] Added unrecognized word \"{}\"".format(unrecognized_word))
                self._unrecognized_words.append(unrecognized_word)
            return False
        return True
