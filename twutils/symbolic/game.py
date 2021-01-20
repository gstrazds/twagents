from symbolic import event
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


# Actions that are dissallowed in any game
ILLEGAL_ACTIONS = ['restart', 'verbose', 'save', 'restore', 'score', 'quit', 'moves']


class GameInstance:
    def __init__(self, kg=None, gt=None, logger=None):  # kg : knowledge_graph.KnowledgeGraph
        self._logger = logger
        self.event_stream = event.EventStream(logger)
        self._unrecognized_words = ILLEGAL_ACTIONS[:]  #makes a copy of list
        self.kg = None
        self.gt = None
        self.set_knowledge_graph(kg, groundtruth=False)
        self.set_knowledge_graph(gt, groundtruth=True)

    def set_knowledge_graph(self, graph, groundtruth=False):  # graph: knowledge_graph.KnowledgeGraph):
        old_graph = self.gt if groundtruth else self.kg
        if old_graph is not None:
            old_graph.event_stream = None
        if graph is not None:
            graph.event_stream = self.event_stream
            assert graph.groundtruth == groundtruth
            graph.groundtruth = groundtruth
            graph.set_logger(self._logger)
        if groundtruth:
            self.gt = graph
        else:
            self.kg = graph

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
                self._logger.debug("[UTIL] Added unrecognized word \"{}\"".format(unrecognized_word))
                self._unrecognized_words.append(unrecognized_word)
            return False
        return True
