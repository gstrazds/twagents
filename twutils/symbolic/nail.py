import os
import logging
from symbolic.game import GameInstance
from symbolic import gv
from symbolic.decision_modules import Idler, Examiner, Interactor, Navigator, Hoarder #, YesNo, YouHaveTo, Darkness
from symbolic.decision_modules import GTNavigator, GTEnder
# from symbolic.knowledge_graph import *
from symbolic.event import NewTransitionEvent, GroundTruthComplete
from symbolic.entity import Entity
from symbolic.location import Location
from symbolic import knowledge_graph
from symbolic.action import *
# from twutils.twlogic import DIRECTION_RELATIONS

DIRECTION_ACTIONS = {
        'north_of': GoNorth,
        'south_of': GoSouth,
        'east_of': GoEast,
        'west_of': GoWest}

LOCATION_RELATIONS = ['at', 'in', 'on']


def entity_type_for_twvar(vartype):
    if vartype in Entity.entity_types:
        return vartype  # 1:1 mapping, for now
    return None


def find_door(fact_list, from_room, to_room):  # return name of door
    for fact in fact_list:
        if fact.name == 'link'\
        and fact.arguments[0].name == from_room.name \
        and fact.arguments[2].name == to_room.name:
            return fact.arguments[1].name
    return None

def add_attributes_for_type(entity, twvartype):
    if twvartype == 't' \
    or twvartype == 'P' \
    or twvartype == 'I' \
    or twvartype == 'r':
        pass
    elif twvartype == 'o':
        entity.add_attribute(Portable)
    elif twvartype == 'f':
        entity.add_attribute(Portable)
        entity.add_attribute(Edible)
        entity.add_attribute(Cutable)
        entity.add_attribute(Cookable)
    elif twvartype == 'c':
        entity.add_attribute(Container)
        entity.add_attribute(Openable)
    elif twvartype == 's':
        entity.add_attribute(Support)
    elif twvartype == 'k':
        entity.add_attribute(Portable)
    elif twvartype == 'd':
        entity.add_attribute(Openable)
        # entity.add_attribute(Lockable)
    elif twvartype == 'oven':
        entity.add_attribute(Container)
        entity.add_attribute(Openable)
    elif twvartype == 'stove':
        entity.add_attribute(Support)
    elif twvartype == 'bbq':
        entity.add_attribute(Cooker)
    elif twvartype == 'toaster':
        entity.add_attribute(Cooker)
    elif twvartype == 'meal':
        entity.add_attribute(Preparable)
    elif twvartype == 'ingredient':
        pass
    elif twvartype == 'slot':
        pass
    elif twvartype == 'RECIPE':
        pass
    else:
        print("Warning -- unexpected variable type:", twvartype, entity)

def add_attributes_for_predicate(entity, predicate, entity2=None):
    if predicate == 'sharp':
        entity.add_attribute(Sharp)
    elif predicate == 'closed':
        entity.add_attribute(Openable)
        entity.state.close()
    elif predicate == 'open':
        entity.add_attribute(Openable)
        entity.state.open()
    elif predicate == 'locked':
        entity.add_attribute(Lockable)
        entity.state.locked()
    elif predicate == 'in':
        entity.add_attribute(Portable)
        if entity2:
            entity2.add_attribute(Container)
    elif predicate == 'on':
        entity.add_attribute(Portable)
        if entity2:
            entity2.add_attribute(Support)
    elif predicate == 'raw':
        entity.add_attribute(Cookable)
    elif predicate == 'inedible':
        entity.del_attribute(Edible)
    elif predicate == 'edible':
        entity.del_attribute(Edible)
    elif predicate == 'drinkable':
        entity.del_attribute(Drinkable)
    elif predicate == 'cookable':
        entity.add_attribute(Cookable)
    elif predicate == 'cooked' \
      or predicate == 'fried' \
      or predicate == 'baked' \
      or predicate == 'toasted' \
      or predicate == 'roasted':
        entity.add_attribute(Edible)
        entity.add_attribute(Cookable)
        entity.state.cook(cooked_state=predicate)
    elif predicate == 'needs_cooking':
        entity.add_attribute(Cookable)
        entity.state.not_cooked()
    elif predicate == 'uncut':
        entity.add_attribute(Cutable)
    elif predicate == 'cuttable':
        entity.add_attribute(Cutable)
    elif predicate == 'chopped' \
         or predicate == 'sliced' \
         or predicate == 'diced' \
         or predicate == 'minced':
        entity.add_attribute(Cutable)
        #entity.state.cut(cut_state=predicate)
    else:
        print("Warning -- unexpected predicate:", predicate, entity)



class NailAgent():
    """
    NAIL Agent: Navigate, Acquire, Interact, Learn

    NAIL has a set of decision modules which compete for control over low-level
    actions. Changes in world-state and knowledge_graph stream events to the
    decision modules. The modules then update how eager they are to take control.

    """
    def __init__(self, seed, env, rom_name, output_subdir='.'):
        self.setup_logging(rom_name, output_subdir)
        gv.rng.seed(seed)
        gv.dbg("RandomSeed: {}".format(seed))
        observed_knowledge_graph  = knowledge_graph.KnowledgeGraph()
        groundtruth_graph = knowledge_graph.KnowledgeGraph(groundtruth=True)
        self.gi = GameInstance(observed_knowledge_graph, groundtruth_graph)
        # self.knowledge_graph.__init__() # Re-initialize KnowledgeGraph
        # gv.event_stream.clear()
        self.gt_nav = GTNavigator(False)
        self.modules = [
                        GTEnder(True),
                        self.gt_nav,
                        #Explorer(True),
                        # Navigator(True),
                        Idler(True),
                        Examiner(True), Hoarder(True),
                        # Interactor(True),
                        # YesNo(True), YouHaveTo(True), Darkness(True)
                        ]
        self.active_module    = None
        self.action_generator = None
        self.first_step       = True
        self._valid_detector  = None  #LearnedValidDetector()
        if env and rom_name:
            self.env = env
            self.step_num = 0

    def setup_logging(self, rom_name, output_subdir):
        """ Configure the logging facilities. """
        for handler in logging.root.handlers[:]:
            handler.close()
            logging.root.removeHandler(handler)
        self.logpath = os.path.join(output_subdir, 'nail_logs')
        if not os.path.exists(self.logpath):
            os.mkdir(self.logpath)
        self.kgs_dir_path = os.path.join(output_subdir, 'kgs')
        if not os.path.exists(self.kgs_dir_path):
            os.mkdir(self.kgs_dir_path)
        self.logpath = os.path.join(self.logpath, rom_name)
        logging.basicConfig(format='%(message)s', filename=self.logpath+'.log',
                            level=logging.DEBUG, filemode='w')

    def elect_new_active_module(self):
        """ Selects the most eager module to take control. """
        most_eager = 0.
        for module in self.modules:
            eagerness = module.get_eagerness(self.gi)
            if eagerness >= most_eager:
                self.active_module = module
                most_eager = eagerness
        print("elect_new_active_module:", "[NAIL](elect): {} Eagerness: {}".format(type(self.active_module).__name__, most_eager))
        gv.dbg("[NAIL](elect): {} Eagerness: {}".format(type(self.active_module).__name__, most_eager))
        self.action_generator = self.active_module.take_control(self.gi)
        self.action_generator.send(None)  # handshake with initial argless yield

    def generate_next_action(self, observation):
        """Returns the action selected by the current active module and
        selects a new active module if the current one is finished.

        """
        next_action = None
        while not next_action:
            try:
                next_action = self.action_generator.send(observation)
                print("[NAIL] (generate_next_action): ({}) -> |{}|".format(type(self.active_module).__name__, next_action))
            except StopIteration:
                self.consume_event_stream()
                self.elect_new_active_module()
        return next_action.text()

    def consume_event_stream(self):
        """ Each module processes stored events then the stream is cleared. """
        for module in self.modules:
            module.process_event_stream(self.gi)
        self.gi.event_stream.clear()

    def _get_gt_location(self, roomname, create_if_notfound=False):
        locations = self.gi.gt.locations_with_name(roomname)
        if locations:
            assert len(locations) == 1
            return locations[0]
        elif create_if_notfound:
            new_loc = Location(roomname)
            ev = self.gi.gt.add_location(new_loc)
            # DISCARD NewlocationEvent -- self.gi.event_stream.push(ev)
            print("created new GT Location:", new_loc)
            return new_loc
        return None

    def _get_gt_entity(self, name, locations=None, holder=None, entitytype=None, create_if_notfound=False):
        if create_if_notfound:
            assert locations is not None
        entities = set()
        if holder:
            for e in holder._entities:
                if e.has_name(name):
                    entities.add(e)
        elif locations:
            for l in locations:
                for e in l.entities:
                    if e.has_name(name):
                        entities.add(e)
        if not entities:  # none found
            entities = self.gi.gt.entities_with_name(name, entitytype=entitytype)
        if entities:
            if len(entities) == 1:
                return list(entities)[0], None
            else:
                found = None
                for e in entities:
                    if entitytype is None or e._type == entitytype:
                        found = e
                if found:
                    return found, None
        if create_if_notfound:
            new_entity = Entity(name, locations[0], type=entitytype)
            if holder:
                ev = holder.add_entity(new_entity)
            else:
                ev = locations[0].add_entity(new_entity)
                if len(locations) > 0:
                    for l in locations[1:]:
                        l.add_entity(new_entity)
            # DISCARD NewEntityEvent -- self.gi.event_stream.push(ev)
            print("created new GT Entity:", new_entity)
            return new_entity, ev
        return None, None

    def _add_obj_to_obj(self, fact, player_loc):
        o = fact.arguments[0]
        h = fact.arguments[1]
        if o.name.startswith('~') or h.name.startswith('~'):
            print("_add_obj_to_obj: SKIPPING FACT", fact)
            return None, None
        if h.name == 'I':  # Inventory
            holder = None  #self.gi.gt.inventory
            loc = self.gi.gt.inventory  #player_loc
        else:
            holder, _ = self._get_gt_entity(h.name, entitytype=entity_type_for_twvar(h.type), locations=None,
                                         create_if_notfound=False)
            loc = holder._init_loc if holder is not None else None
        if loc:
            loc = [loc]
        else:
            print("WARNING! NO LOCATION FOR HOLDER while adding GT Object {} {}".format(fact, holder))
            loc = None

    #NOTE TODO: handle objects that have moved from to or from Inventory
        obj, ev = self._get_gt_entity(o.name, entitytype=entity_type_for_twvar(o.type), locations=loc, holder=holder,
                                  create_if_notfound=True)
        #add entity to entity (inventory is of type 'location', adding is done by create_if_notfound)
        # if holder:
        #     holder.add_entity(obj)
        if ev:
            print("ADDED NEW GT Object {} :{}: {}".format(obj, fact.name, holder))
        else:
            print("FOUND GT Object {} :{} {}".format(obj, fact.name, holder))
        return obj, holder

    def gt_navigate(self, roomname):
        dest_loc = self._get_gt_location(roomname, create_if_notfound=False)
        assert dest_loc is not None
        self.gt_nav.set_goal(dest_loc, self.gi)
        # self.elect_new_active_module()

    def set_ground_truth(self, gt_facts):
        # print("GROUND TRUTH")
        # sort into separate lists to control the order in which facts get processed
        player_loc = None
        door_facts = []
        at_facts = []
        on_facts = []
        in_facts = []
        other_facts = []
        for fact in gt_facts:
            a0 = fact.arguments[0]
            a1 = fact.arguments[1] if len(fact.arguments) > 1 else None
            if fact.name == 'link':
                door_facts.append(fact)
            elif fact.name == 'at':
                at_facts.append(fact)
                if a0.type == 'P' and a1.type == 'r':
                    player_loc = self._get_gt_location(a1.name, create_if_notfound=True)
            elif fact.name == 'on':
                on_facts.append(fact)
            elif fact.name == 'in':
                in_facts.append(fact)
            elif fact.name in DIRECTION_ACTIONS:
                if a0.type == 'r' and a1.type == 'r':
                    # During this initial pass we create locations and connections among them
                    # print('++CONNECTION:', fact)
                    loc0 = self._get_gt_location(a0.name, create_if_notfound=True)
                    loc1 = self._get_gt_location(a1.name, create_if_notfound=True)
                    # door_name = find_door(gt_facts, a1, a0)
                    # if door_name:
                    #     door = self._get_gt_entity(door_name, entitytype=gv.DOOR, locations=[loc1, loc0], create_if_notfound=True)
                    # else:
                    #     door = None
                    door = None  # will add info about doors later
                    new_connection = knowledge_graph.Connection(loc1, DIRECTION_ACTIONS[fact.name], loc0, doorway=door)
                    self.gi.gt.add_connection(new_connection, self.gi)  # does nothing if connection already present
                elif a0.type == 'd' and a1.type == 'r':
                    # print("\t\tset_ground_truth: TODO door fact -- ", fact)
                    pass
                else:
                    # print("--IGNORING:", fact)
                    pass
            else:
                other_facts.append(fact)
        # 2nd pass, add doors to connections
        for fact in door_facts:
            assert len(fact.arguments) == 3
            r0 = fact.arguments[0]
            d = fact.arguments[1]
            r1 = fact.arguments[2]
            assert r0.type == 'r'
            assert r1.type == 'r'
            assert d.type == 'd'
            loc0 = self._get_gt_location(r0.name, create_if_notfound=False)
            loc1 = self._get_gt_location(r1.name, create_if_notfound=False)
            door, _ = self._get_gt_entity(d.name, entitytype=gv.DOOR, locations=[loc1, loc0], create_if_notfound=True)
            linkpath = self.gi.gt.connections.shortest_path(loc0, loc1)
            assert len(linkpath) == 1
            connection = linkpath[0]
            connection.doorway = door
        for fact in other_facts:
            if fact.name == 'closed' and fact.arguments[0].type == 'd':
                doorname = fact.arguments[0].name
                doors = self.gi.gt.entities_with_name(doorname, entitytype=gv.DOOR)
                assert len(doors) == 1
                door = list(doors)[0]
                door.state.close()
        for fact in at_facts:
            o = fact.arguments[0]
            r = fact.arguments[1]
            loc = self._get_gt_location(r.name, create_if_notfound=False)
            if r.type == 'r':
                obj, _ = self._get_gt_entity(o.name, entitytype=entity_type_for_twvar(o.type), locations=[loc], create_if_notfound=True)
            else:
                gv.dbg("WARNING -- SET GROUND TRUTH: unexpected location for at(o,l): {}".format(r))
            add_attributes_for_type(obj, o.type)

    #NOTE: the following assumes that objects are not stacked on top of other objects which are on or in objects
    # and similarly, that "in" relations are not nested.
    #TODO: this should be generalized to work correctly for arbitrary chains of 'on' and 'in'
        for fact in on_facts:
            o1, o2 = self._add_obj_to_obj(fact, player_loc)
            if o1 and o2:
                add_attributes_for_predicate(o1, 'on', o2)
        for fact in in_facts:
            o1, o2 = self._add_obj_to_obj(fact, player_loc)
            if o1 and o2:
                add_attributes_for_predicate(o1, 'in', o2)
        if player_loc:
            if self.gi.gt.set_player_location(player_loc, self.gi):
                print("CHANGED GT player location:", player_loc)

        for fact in other_facts:
            predicate = fact.name
            if predicate == 'cooking_location' \
            or predicate.startswith('ingredient_') \
            or predicate == 'free' \
            or predicate == 'base' \
            or predicate == 'out':
                continue

            a0 = fact.arguments[0]
            if a0.name.startswith('~'):
                continue
            if len(fact.arguments) > 1:
                a1 = fact.arguments[1]
            else:
                a1 = None
            o1, _ = self._get_gt_entity(a0.name, entitytype=entity_type_for_twvar(a0.type))
            if a1:
                o2, _ = self._get_gt_entity(a1.name, entitytype=entity_type_for_twvar(a1.type))
            if o1:
                add_attributes_for_predicate(o1, predicate, entity2=o2)
            else:
                if predicate == 'edible' and a0.name == 'meal':
                    continue
                print("Warning: add_attributes_for_predicate", predicate, "didnt find an entity corresponding to", a0)
        self.gi.event_stream.push(GroundTruthComplete(groundtruth=True))

    def take_action(self, observation, obs_facts=None, gt_facts=None):
        if gt_facts:
            self.set_ground_truth(gt_facts)
        if obs_facts:
            # world = World.from_facts(facts)
            # add obs_facts to our KnowledgeGraph (self.gi.kg)
            pass
        if self.env and getattr(self.env, 'get_player_location', None):
            # Add true locations to the .log file.
            loc = self.env.get_player_location()
            if loc and hasattr(loc, 'num') and hasattr(loc, 'name') and loc.num and loc.name:
                gv.dbg("[TRUE_LOC] {} \"{}\"".format(loc.num, loc.name))

            # Output a snapshot of the kg.
            # with open(os.path.join(self.kgs_dir_path, str(self.step_num) + '.kng'), 'w') as f:
            #     f.write(str(self.knowledge_graph)+'\n\n')
            # self.step_num += 1

        observation = observation.strip()
        if self.first_step:
            gv.dbg("[NAIL] {}".format(observation))
            self.first_step = False
            #GVS# return 'look' # Do a look to get rid of intro text

        if not self.gi.kg.player_location:
            loc = Location(observation)
            ev = self.gi.kg.add_location(loc)
            self.gi.kg.set_player_location(loc, self.gi)
            self.gi.kg._init_loc = loc
            # self.gi.event_stream.push(ev)

        self.consume_event_stream()

        if not self.active_module:
            self.elect_new_active_module()

        next_action = self.generate_next_action(observation)
        return next_action

    def observe(self, prev_obs, action, score, new_obs, terminal):
        """ Observe will be used for learning from rewards. """
#        p_valid = self._valid_detector.action_valid(action, new_obs)
#        gv.dbg("[VALID] p={:.3f} {}".format(p_valid, clean(new_obs)))
#        if kg.player_location:
#            dbg("[EAGERNESS] {}".format(' '.join([str(module.get_eagerness()) for module in self.modules[:5]])))
        self.gi.event_stream.push(NewTransitionEvent(prev_obs, action, score, new_obs, terminal))
        self.gi.action_recognized(action, new_obs)  # Update the unrecognized words
        if terminal:
            self.gi.kg.reset(self.gi)

    def finalize(self):
        # with open(self.logpath+'.kng', 'w') as f:
        #     f.write(str(self.knowledge_graph)+'\n\n')
        pass
