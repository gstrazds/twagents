from symbolic import util


# class EventStream:
#     """
#     An event stream keeps track of incoming events.

#     """
#     def __init__(self, logger):
#         self._stream = []
#         self._logger = logger

#     def push(self, event):
#         if not event.is_groundtruth:
#             self._logger.debug("[LOG]({}) {}".format(type(event).__name__, event.message))
#         self._stream.append(event)

#     def clear(self):
#         del self._stream[:]

#     def read(self):
#         """ Iterate through the events in the stream. """
#         for event in self._stream:
#             yield event


class Event:
    """ Base class for all events. """
    def __init__(self, message, groundtruth=False):
        self.message = message
        self.is_groundtruth = groundtruth


# class NewTransitionEvent(Event):
#     """ Generated whenever an action is taken. """
#     def __init__(self, obs, action, score, new_obs, terminal, groundtruth=False):
#         message = '\"{}\" --> {} Score={}'.format(action, util.clean(new_obs), score)
#         super().__init__(message, groundtruth=groundtruth)
#         self.obs      = obs
#         self.action   = action
#         self.score    = score
#         self.new_obs  = new_obs
#         self.terminal = terminal

class NewLocationEvent(Event):
    """ Generated whenever a new location is discovered. """
    def __init__(self, new_location, groundtruth=False):
        super().__init__(new_location.name, groundtruth=groundtruth)
        self.new_location = new_location

class NewEntityEvent(Event):
    """ Generated whenever a new entity is discovered. """
    def __init__(self, new_entity, groundtruth=False):
        message = "{}: {}".format(new_entity.name, new_entity.description)
        super().__init__(message, groundtruth=groundtruth)
        self.new_entity = new_entity

# class NewActionRecordEvent(Event):
#     """ Generated whenever a new action is applied. """
#     def __init__(self, entity, action_record, result_text, groundtruth=False):
#         message = "{} ==({})==> {}".format(entity, action_record, util.clean(result_text))
#         super().__init__(message, groundtruth=groundtruth)
#         self.entity = entity
#         self.action_record = action_record
#         self.result_text = result_text

class LocationChangedEvent(Event):
    """ Generated whenever the player's location changes. """
    def __init__(self, new_location, groundtruth=False):
        super().__init__(new_location.name, groundtruth=groundtruth)
        self.new_location = new_location

# class NeedToAcquire(Event):
#     """ Generated whenever a decision module determines that one or more specific items should be obtained (into inventory). """
#     def __init__(self, objnames=None, groundtruth=False):
#         self.objnames = objnames
#         message = "Need to acquire: {}".format(objnames)
#         super().__init__(message, groundtruth=groundtruth)

# class NeedToFind(Event):
#     """ Generated whenever a decision module determines that one or more specific items should be located. """
#     def __init__(self, objnames=None, groundtruth=False):
#         self.objnames = objnames
#         message = "Need to find: {}".format(objnames)
#         super().__init__(message, groundtruth=groundtruth)

# class NoLongerNeed(Event):
#     """ Generated whenever a decision module determines that one or more required items are no longer required. """
#     def __init__(self, objnames=None, groundtruth=False):
#         self.objnames = objnames
#         message = "No longer need: {}".format(objnames)
#         super().__init__(message, groundtruth=groundtruth)

# class NeedSequentialSteps(Event):
#     """ Generated whenever a decision module determines a series of sequential actions should be performed. """
#     def __init__(self, steps, groundtruth=False):
#         self.steps = steps
#         message = "Need to perform: {}".format(steps)
#         super().__init__(message, groundtruth=groundtruth)

# class NeedToGoTo(Event):
#     """ Generated when a decision module determines that the agent should navigate to a specific location. """
#     def __init__(self, target_location, groundtruth=False):
#         self.target_location = target_location
#         message = "Need to go to: {}".format(target_location)
#         super().__init__(message, groundtruth=groundtruth)

# class NeedToDo(Event):
#     """ Generated when a decision module determines that the agent needs to perform a specific Task. """
#     def __init__(self, task, groundtruth=False):
#         self.task = task
#         message = "Need to do: {}".format(task)
#         super().__init__(message, groundtruth=groundtruth)
