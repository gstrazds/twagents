from symbolic.action import StandaloneAction
from symbolic.task_modules import SingleActionTask


class SayLocationOfEntityTask(SingleActionTask):
    def __init__(self, entityname: str, description=None, use_groundtruth=False):
        super().__init__(act=StandaloneAction("Unknown"),
                         description=f"SayLocationOfEntityTask[{entityname}]",
                         use_groundtruth=use_groundtruth)
        self._entityname = entityname

    def activate(self, kg, exec):
        if kg.location_of_entity_is_known(self._entityname):
            if super().action:
                assert isinstance(super().action, StandaloneAction)
                if kg.inventory.has_entity_with_name(self._entityname):
                    locname = "inventory"
                else:
                    loc = kg.location_of_entity_with_name(self._entityname)
                    locname = loc.name if loc else "nowhere"
                super().action.verb = f"answer: {locname}"

        return super().activate(kg, exec)


class AnswerFoundTask(SingleActionTask):
    def __init__(self, entityname: str, description=None, use_groundtruth=False):
        super().__init__(act=StandaloneAction("answer: 0"),
                         description=f"AnswerFoundTask[{entityname}]",
                         use_groundtruth=use_groundtruth)
        self._entityname = entityname

    def activate(self, kg, exec):
        if super().action:
            assert isinstance(super().action, StandaloneAction)
            if kg.location_of_entity_is_known(self._entityname):
                found = "1"
            else:
                found = "0"
            super().action.verb = f"answer: {found}"

        return super().activate(kg, exec)