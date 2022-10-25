from typing import Tuple, List
import re    # regular expressions, for finding mentions of entities

# from symbolic.action import *
# from symbolic.entity import ConnectionRelation, Entity, Thing, Location, Person, UnknownLocation, Door
# from symbolic.entity import DOOR, ROOM, PERSON, INVENTORY, CONTAINED_LOCATION, SUPPORTED_LOCATION, NONEXISTENCE_LOC
from symbolic.entity import Entity

def _span_len(span: Tuple):
    return span[1] - span[0]

def _span_overlap(span1: Tuple, span2: Tuple):
    if span1[1] <= span2[0] or span1[0] >= span2[1]:
        return 0  # NO OVERLAP
    if span1[0] >= span2[0] and span1[0] < span2[1]:  # they overlap!
        return min(span2[1], span1[1]) - span1[0]
    return _span_overlap(span2, span1)   # same checks, other ordering

def test_span_overlap():
    overlaps = [
        _span_overlap((0, 1), (0, 1)),    # identical len 1   => 1
        _span_overlap((10, 11), (10, 11)), # identical len 1   => 1
        _span_overlap((2, 4), (2, 4)),     # identical 2 pos   => 2
        _span_overlap((5, 8), (6, 20)),    # partial overlap   => 2
        _span_overlap((6, 20), (5, 8)),    # -> 2
        _span_overlap((5, 10), (4, 10)),   # common end        => 5
        _span_overlap((4, 10), (5, 10)),   # command end       => 5
        _span_overlap((5, 10), (5, 11)),   # common start      => 5
        _span_overlap((5, 11), (5, 10)),   # common start      => 5
    ]
    if overlaps != [1,1,2,2,2,5,5,5,5]:
        assert overlaps == [1, 1, 2, 2, 2, 5, 5, 5, 5]
        return False

    non_overlaps = [
        _span_overlap((5, 7), (8, 10)),  # disjoint
        _span_overlap((8, 10), (5, 7)),  # disjoint
        _span_overlap((0, 1), (1, 2)),  # adjacent 1 pos
        _span_overlap((1, 2), (0, 1)),  # adjacent 1 pos
        _span_overlap((10, 11), (11, 12)),  # adjacent 1 pos
        _span_overlap((11, 12), (12, 11)),  # adjacent 1 pos
        _span_overlap((10, 15), (15, 20)),  # adjacent longer
        _span_overlap((15, 20), (10, 15)),  # adjacent longer
    ]
    _zero_overlaps = [0] * 8
    if non_overlaps != _zero_overlaps:
        assert non_overlaps == _zero_overlaps
        return False
    return True

test_span_overlap()   # module won't load if this test fails!!!


class MentionIndex:
    def __init__(self, observ_str:str):
        self._observ_str = observ_str
        self.mentions = {}  # dict like {name: List[Tuple(start,end)]}

    def is_subspan_of_existing(self, span, name=''):
        _my_len = _span_len(span)
        for k in self.mentions:
            span_list = self.mentions[k]
            for span0 in span_list:
                overlap = _span_overlap(span0, span)
                if overlap:
                    if _my_len == overlap:   # full inclusion
                        print(f"New span for [{name}]{span} is included in [{k}]{span0}")
                        return True
                    if _my_len <= _span_len(span0):
                        if span[0] < span0[0] or span[1] > span0[1]:
                            print(f"!!! WARNING: ignoring SHORTER new span for [{name}]{span} OVERLAPS [{k}]{span0}")
                        return True
                    else:  # _my_len > _span_len(span0):
                        #print(f"WARNING!!! LONGER new span for [{name}]{span} OVERLAPS [{k}]{span0} => will unindex [{k}]{span0}")
                        pass
        return False

    def remove_subspans_of(self, span, name=None):
        """ if any existing spans are (shorter) subspans of span, remove them from the index"""
        _my_len = _span_len(span)
        for k in self.mentions:
            span_list = self.mentions[k]
            for span0 in list(span_list):  # iterate through a copy of the list
                overlap = _span_overlap(span0, span)
                if overlap:
                    if _my_len <= _span_len(span0):  # full inclusion or new is shorter
                        assert False, f"SHOULD ALREADY BE FILTERED OUT: shorter new span [{name}]{span} OVERLAPS [{k}]{span0}"
                        continue
                    else:  # _my_len > _span_len(span0):
                        print(f"UNINDEXING [{k}]{span0} because of LONGER new span: [{name}]{span}")
                        span_list.remove(span0)

    def get_first_mention(self, name: str) -> int:
        if name in self.mentions:
            return self.mentions[name][0][0]   # start position of first mention
        return -1   # mention not found

    def add_mention(self, name: str, span: Tuple):
        """ Adds a mention to the index.
         In case of an overlap with other mentions
          (which should be either a full inclusion or a superset),
         the shorter span is removed from the index and the longer is kept"""
        # print(f"add_mention({name}, {span}")
        if self.is_subspan_of_existing(span, name=name):
            return None
        else:
            self.remove_subspans_of(span, name)
            if name in self.mentions:
                if span not in self.mentions[name]:
                    self.mentions[name].append(span)
                    self.mentions[name] = sorted(self.mentions[name])  # maintain sorted ordering (short lists)
            else:
                self.mentions[name] = [span]
        return self.mentions[name][0]  # return span of first mention

    def index_mentions(self, name: str):  # adds all mentions
        # print("INDEX MENTIONS of:", name)
        matches = re.finditer(name, self._observ_str)
        for span in [m.span() for m in matches]:
            # print(name, ":", span)
            self.add_mention(name, span)
        # print(self.mentions)

    def sort_facts_by_first_mention(self, facts_list):   # returns sorted facts_list
        zip_list = []
        for fact in facts_list:
            pass
        return facts_list

    def sort_entity_list_by_first_mention(self, entities):   # returns sorted list of entity names (or entities)
        zip_list = []
        # print("PRE: sort_entity_list:", self.mentions)
        for entity in entities:
            if isinstance(entity, Entity):
                name = entity.name
            else:    # we can also sort name strings
                name = str(entity)

            mention_start = self.get_first_mention(name)
            if mention_start >= 0:
                zip_list.append((mention_start, entity))
            else:
                if isinstance(entity, Entity):
                    print("WARNING: SORT ENTITY LIST BY MENTION DIDN'T FIND", entity, entity.parent, entity.location)
                    # assert False
                else:
                    print("WARNING: SORT ENTITY LIST BY MENTION DIDN'T FIND str:", entity)
                zip_list.append((100000+len(zip_list), entity))
        # print("POST: sort_entity_list:", self.mentions)
        zip_list = sorted(zip_list)
        sorted_list = [entity for (start, entity) in zip_list]
        return sorted_list

    def subst_names(self, name_ids: List[Tuple[str, str]]) -> str:
        """ Returns a copy of the observation text with variable ids
         in place of entity names"""
        edited_obs = str(self._observ_str)
        # replace longer names first
        sorted_list = list(sorted(name_ids, key=lambda tup: len(tup[0]), reverse=True))
        for varname, varid in sorted_list:
            if varname in self.mentions:
                print(f"Replacing {varname}<-{varid} for spans {self.mentions[name]}")
                edited_obs = edited_obs.replace(varname, varid)
                print("\t-> ", edited_obs)
        return edited_obs