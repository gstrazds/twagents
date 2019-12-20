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
    def __init__(self, kg=None, gt=None):  # kg : knowledge_graph.KnowledgeGraph
        self.event_stream = event.EventStream()
        self._unrecognized_words = gv.ILLEGAL_ACTIONS[:]  #makes a copy of list
        #
        self.kg = kg
        self.gt = gt  # ground truth knowledge graph

    def entity_at_location(self, entity, location):
        if location.add_entity(entity):
            ev = event.NewEntityEvent(entity)
            self.event_stream.push(ev)

    # def entity_at_entity(self, entity1, entity2):
    #     if entity1.add_entity(entity2):
    #         ev = event.NewEntityEvent(entity2)
    #         self.event_stream.push(ev)

    def act_on_entity(self, action, entity, p_valid, result_text):
        if entity.add_action_record(action, p_valid, result_text):
            ev = event.NewActionRecordEvent(entity, action, result_text)
            self.event_stream.push(ev)

    # def add_entity_attribute(self, entity, attribute, groundtruth=False):
    #     if entity.add_attribute(attribute):
    #         if not groundtruth:
    #             ev = event.NewAttributeEvent(entity, attribute, groundtruth=groundtruth)
    #             self.event_stream.push(ev)

    def move_entity(self, entity, origin, dest, groundtruth=False):
        """ Moves entity from origin to destination. """
        assert origin.has_entity(entity), \
            "Can't move entity {} that isn't present at origin {}" \
            .format(entity, origin)
        origin.del_entity(entity)
        if not dest.add_entity(entity):
            msg = f"Unexpected: already at target location {dest}.add_entity(entity={entity}) origin={origin})"
            print("!!!WARNING: "+msg)
            assert False, msg
        if not groundtruth:
            self.event_stream.push(event.EntityMovedEvent(entity, origin, dest, groundtruth=groundtruth))

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
