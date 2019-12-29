# import os, sys
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from symbolic.event import *
from symbolic.action import *
from symbolic.entity import Entity, Thing, DOOR, ROOM
from symbolic.entity import Location, Person, UnknownLocation, Door

DIRECTION_ACTIONS = {
        'north_of': GoNorth,
        'south_of': GoSouth,
        'east_of': GoEast,
        'west_of': GoWest}

LOCATION_RELATIONS = ['at', 'in', 'on']


def get_attributes_for_type(twvartype:str):
    attrib_list = []
    if twvartype == 't' \
    or twvartype == 'P' \
    or twvartype == 'I' \
    or twvartype == 'r':
        pass
    elif twvartype == 'o':
        attrib_list.append(Portable)
    elif twvartype == 'f':
        attrib_list.append(Portable)
        attrib_list.append(Edible)
        attrib_list.append(Cutable)
        attrib_list.append(Cookable)
    elif twvartype == 'c':
        attrib_list.append(Container)
        attrib_list.append(Openable)
    elif twvartype == 's':
        attrib_list.append(Support)
    elif twvartype == 'k':
        attrib_list.append(Portable)
    elif twvartype == 'd':
        attrib_list.append(Openable)
        # attrib_list.append(Lockable)
    elif twvartype == 'oven':
        attrib_list.append(Container)
        attrib_list.append(Openable)
    elif twvartype == 'stove':
        attrib_list.append(Support)
    elif twvartype == 'bbq':
        attrib_list.append(Cooker)
    elif twvartype == 'toaster':
        attrib_list.append(Cooker)
    elif twvartype == 'meal':
        attrib_list.append(Preparable)
    elif twvartype == 'ingredient':
        pass
    elif twvartype == 'slot':
        pass
    elif twvartype == 'RECIPE':
        pass
    else:
        print("Warning -- get_attributes_for_type() unexpected variable type:", twvartype)
    return attrib_list


def get_attributes_for_predicate(predicate, entity2):
    attrib_list = []
    if predicate == 'sharp':
        attrib_list.append(Sharp)
    elif predicate == 'closed':
        attrib_list.append(Openable)
    elif predicate == 'open':
        attrib_list.append(Openable)
    elif predicate == 'locked':
        attrib_list.append(Lockable)
    elif predicate == 'in':
        attrib_list.append(Portable)
        if entity2:
            entity2.add_attribute(Container)
    elif predicate == 'on':
        attrib_list.append(Portable)
        if entity2:
            entity2.add_attribute(Support)
    elif predicate == 'raw':
        attrib_list.append(Cookable)
    # elif predicate == 'inedible':
    #     entity.del_attribute(Edible)
    elif predicate == 'edible':
        attrib_list.append(Edible)
    elif predicate == 'drinkable':
        attrib_list.append(Drinkable)
    elif predicate == 'cookable':
        attrib_list.append(Cookable)
    elif predicate == 'cooked' \
      or predicate == 'fried' \
      or predicate == 'baked' \
      or predicate == 'toasted' \
      or predicate == 'grilled' \
      or predicate == 'roasted':
        attrib_list.append(Edible)
        attrib_list.append(Cookable)
    elif predicate == 'needs_cooking':
        attrib_list.append(Cookable)
    elif predicate == 'uncut':
        attrib_list.append(Cutable)
    elif predicate == 'cuttable':
        attrib_list.append(Cutable)
    elif predicate == 'chopped' \
     or predicate == 'sliced' \
     or predicate == 'diced' \
     or predicate == 'minced':
        attrib_list.append(Cutable)
    else:
        print("Warning -- get_attributes_for_predicate() unexpected predicate:", predicate)
    return attrib_list


def add_attributes_for_type(entity, twvartype):
    attrib_list = get_attributes_for_type(twvartype)
    for attrib in attrib_list:
        entity.add_attribute(attrib)
    if twvartype == 'd':  # DOOR   Hard-coded specific knowlege about doors (they can potentially be opened)
        if not entity.state.openable:
            entity.state.add_state_variable('openable', 'is_open', '')
            # Indeterminate value until we discover otherwise (via add_attributes_for_predicate)


def add_attributes_for_predicate(entity, predicate, entity2=None):
    attrib_list = get_attributes_for_predicate(predicate, entity2)
    for attrib in attrib_list:
        entity.add_attribute(attrib)

    # inverse relationships
    if predicate == 'in':
        assert entity2 is not None
        entity2.add_attribute(Container)
    elif predicate == 'on':
        assert entity2 is not None
        entity2.add_attribute(Support)

    if predicate == 'inedible':
        entity.del_attribute(Edible)

    # set entity state
    if predicate == 'closed':
        entity.state.close()
    elif predicate == 'open':
        entity.state.open()
    elif predicate == 'locked':
        entity.state.locked()
    elif predicate == 'cooked' \
      or predicate == 'fried' \
      or predicate == 'baked' \
      or predicate == 'toasted' \
      or predicate == 'grilled' \
      or predicate == 'roasted':
        entity.state.cook(cooked_state=predicate)
    elif predicate == 'needs_cooking':
        entity.state.not_cooked()
    elif predicate == 'uncut':
        entity.state.not_cut()
    elif predicate == 'cuttable':
        entity.add_attribute(Cutable)
    elif predicate == 'chopped' \
     or predicate == 'sliced' \
     or predicate == 'diced' \
     or predicate == 'minced':
        entity.state.cut(cut_state=predicate)


def entity_type_for_twvar(vartype):
    if vartype in Entity.entity_types:
        return vartype  # 1:1 mapping, for now
    return None


# def find_door(fact_list, from_room, to_room):  # return name of door
#     for fact in fact_list:
#         if fact.name == 'link'\
#         and fact.arguments[0].name == from_room.name \
#         and fact.arguments[2].name == to_room.name:
#             return fact.arguments[1].name
#     return None


class KnowledgeGraph:
    """
    Knowledge Representation consists of visisted locations.

    """
    def __init__(self, event_stream, groundtruth=False):
        self._unknown_location   = UnknownLocation()
        self._player             = Person(name="You",
                                          description="The Protagonist",
                                          location=self._unknown_location)
        self._locations          = []   # a set of top-level locations (rooms)
        # self._inventory          = self._player.inventory
        self._connections        = ConnectionGraph()  # navigation links connecting self._locations
        self.event_stream        = event_stream
        self.groundtruth         = groundtruth   # bool: True means this is the Ground Truth knowledge graph

    @property
    def locations(self):
        return self._locations

    def add_location(self, new_location: Location) -> NewLocationEvent:
        """ Adds a new location object and broadcasts a NewLocation event. """
        self._locations.append(new_location)
        return NewLocationEvent(new_location, groundtruth=self.groundtruth)

    def locations_with_name(self, location_name):
        """ Returns all locations with a particular name. """
        return [loc for loc in self._locations if loc.has_name(location_name)]

    @property
    def player_location(self):
        return self._player.location

    # @player_location.setter
    def set_player_location(self, new_location):
        """ Changes player location and broadcasts a LocationChangedEvent. """
        prev_location = self._player.location
        if new_location == prev_location:
            return False
        if new_location:
            new_location.visit()
            self.event_stream.push(LocationChangedEvent(new_location, groundtruth=self.groundtruth))
        self._player.location = new_location
        return True

    @property
    def inventory(self):
        return self._player.inventory

    @property
    def connections(self):
        return self._connections

    def add_connection(self, new_connection):
        """ Adds a connection object.
         Does nothing if a similar connection already exists"""
        self._connections.add(new_connection, self)

    def reset(self):
        """Returns the knowledge_graph to a state resembling the start of the
        game. Note this does not remove discovered objects or locations. """
        kg = self
        self.inventory.reset(kg)
        self._player.reset(kg)
        for location in self.locations:
            location.reset(kg)

    def __str__(self):
        s = "Knowledge Graph{}\n".format('[GT]' if self.groundtruth else '')
        if self.player_location == None:
            s += "PlayerLocation: None"
        else:
            s += "PlayerLocation: {}".format(self.player_location.name)
        s += "\n" + str(self.inventory)
        s += "\nKnownLocations:"
        if self._locations:
            for loc in self._locations:
                s += "\n" + loc.to_string("  ")
                outgoing = self.connections.outgoing(loc)
                if outgoing:
                    s += "\n    Connections:"
                    for con in outgoing:
                        s += "\n      {} --> {}".format(con.action, con.to_location.name)
        return s

    def entities_with_name(self, entityname, entitytype=None):
        """ Returns all entities with a particular name. """
        ret = set()
        for loc in (self._locations + [self.inventory]):
            e = loc.get_entity_by_name(entityname)
            if e and (entitytype is None or e._type == entitytype):
                ret.add(e)
        # if len(ret) == 0:   # check also on or in other entities (just one level deep)
        #     for l in (self._locations + [self.inventory]):
        #         for e in l.entities:
        #             for e2 in e._entities:
        #                 if e2 and e2.has_name(entityname) and (entitytype is None or e2._type == entitytype):
        #                     ret.add(e2)
        if len(ret) == 0:  # check to see if entity has been previously mentioned, but not yet found
            e = self._unknown_location.get_entity_by_name(entityname)
            if e and (entitytype is None or e._type == entitytype):
                ret.add(e)
        return ret

    def where_is_entity(self, entityname, entitytype=None, allow_unknown=True):
        """ Returns a set of locations where an entity with a specific name can be found """
        ret = set()
        search_locations = self._locations + [self.inventory]
        if allow_unknown:
            search_locations.append(self._unknown_location)
        for loc in search_locations:
            e = loc.get_entity_by_name(entityname)
            if e and (entitytype is None or e._type == entitytype):
                ret.add(loc)
        return ret

    def location_of_entity(self, entityname, entitytype=None, allow_unknown=False):
        """ Returns a single location where an entity with a specific name can be found """
        location_set = self.where_is_entity(entityname, entitytype=entitytype, allow_unknown=allow_unknown)
        # print(f"DEBUG location_of_entity({entityname}:{entitytype}) => {location_set}")
        if not allow_unknown:
            location_set.discard(self._unknown_location)
        if location_set:
            if len(location_set) > 1:
                print(f"WARNING: multiple locations for <{entityname}>: {location_set}")
            return location_set.pop()  # choose one element from the set
        return None

    def get_containing_entity(self, entity):
        """ Returns container (or support) where an entity with a specific name can be found """
        for loc in self._locations:
            for ce in loc.entities:
                if ce.has_entity(entity):
                    return ce
        # if len(ret) == 0:   # check also on or in other entities (just one level deep)
        #     for l in (self._locations + [self.inventory]):
        #         for e in l.entities:
        #             for e2 in e._entities:
        #                 if e2 and e2.has_name(entityname) and (entitytype is None or e2._type == entitytype):
        #                     ret.add(l)
        return None

    def entity_at_location(self, entity, location):
        if location.add_entity(entity):
            if self.event_stream and not self.groundtruth:
                ev = NewEntityEvent(entity)
                self.event_stream.push(ev)

    # def entity_at_entity(self, entity1, entity2):
    #     if entity1.add_entity(entity2):
    #         ev = event.NewEntityEvent(entity2)
    #         self.event_stream.push(ev)

    def act_on_entity(self, action, entity, rec: ActionRec):
        if entity.add_action_record(action, rec) and rec.p_valid > 0.5 and not self.groundtruth:
            ev = NewActionRecordEvent(entity, action, rec.result_text)
            self.event_stream.push(ev)

    # def add_entity_attribute(self, entity, attribute, groundtruth=False):
    #     if entity.add_attribute(attribute):
    #         if not groundtruth:
    #             ev = event.NewAttributeEvent(entity, attribute, groundtruth=groundtruth)
    #             self.event_stream.push(ev)
    def action_at_current_location(self, action, p_valid, result_text):
        loc = self.player_location
        loc.action_records[action] = ActionRec(p_valid, result_text)
        if not self.groundtruth:
            ev = NewActionRecordEvent(loc, action, result_text)
            self.event_stream.push(ev)

    def get_location(self, roomname, create_if_notfound=False, type=ROOM):
        locations = self.locations_with_name(roomname)
        if locations:
            assert len(locations) == 1
            return locations[0]
        elif create_if_notfound:
            new_loc = Location(name=roomname, type=type)
            ev = self.add_location(new_loc)
            # if self.groundtruth: DISCARD NewLocationEvent else gi.event_stream.push(ev)
            if self.groundtruth:
                new_loc._discovered = True    # HACK for GT logic
            else:  # not self.groundtruth:
                assert ev, "Adding a newly created Location should return a NewLocationEvent"
                self.event_stream.push(ev)
            print("created new Location:", new_loc)
            return new_loc
        print("LOCATION NOT FOUND:", roomname)
        return None

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
            self.event_stream.push(EntityMovedEvent(entity, origin, dest, groundtruth=groundtruth))

    def maybe_move_entity(self, entity, locations=None):
        assert locations
        if locations:   # ? and create_if_notfound:
            # might need to move it from wherever it was to its new location

            prev_loc_set = self.where_is_entity(entity.name, entitytype=entity._type, allow_unknown=True)
            loc_set = set(locations)
            if prev_loc_set != loc_set:
                print(f"get_entity() WARNING - MOVING {entity} to {loc_set}")
                if len(prev_loc_set) != len(loc_set):
                    print("WARNING: CAN'T HANDLE len(prev_loc_set) != len(loc_set):", entity, prev_loc_set, locations)
                elif prev_loc_set:  # available information about location object seems to have changed
                    if len(prev_loc_set) == 2 and len(loc_set) == 2:
                        if entity._type != DOOR:
                            print("WARNING: expected type==DOOR:", entity)
                        new_locs = loc_set - prev_loc_set   # set.difference()
                        prev_locs = prev_loc_set - loc_set
                        if len(new_locs) == 1 and len(prev_locs) == 1:
                            prev_loc_set = set([self._unknown_location])
                            loc_set = new_locs
                            # drop through to if len(prev_loc_set) == 1 case, below...
                        else:
                            print(f"WARNING: UNEXPECTED previous locations for {entity} prev:{prev_loc_set} "
                                  f"-> new:{new_locs}; {locations}")

                    if len(prev_loc_set) == 1:  # and len(prev_loc_set) == 1:
                        assert len(loc_set) == 1
                        loc_prev = prev_loc_set.pop()
                        loc_new = loc_set.pop()
                        # TODO: here we are assuming exactly one found entity and one location
                        if loc_prev != self._unknown_location:
                            print(f"WARNING: KG.get_entity() triggering move_entity(entity={entity},"
                                  f" dest={loc_new}, origin={loc_prev})")
                        if loc_new == self._unknown_location:
                            print(f"ERROR: not moving {entity} to UnknownLocation")
                        else:
                            self.move_entity(entity, loc_prev, loc_new, groundtruth=self.groundtruth)
                    else:
                        print("WARNING: CAN'T HANDLE multiple locations > 2", prev_loc_set, locations)


    def get_entity(self, name, entitytype=None):
        entities = set()
        if not entities:  # none found
            entities = self.entities_with_name(name, entitytype=entitytype)
        if entities:
            if len(entities) == 1:
                return entities.pop()
            else:
                found = None
                for e in entities:
                    if entitytype is None or e._type == entitytype:
                        found = e
                return found
        return None

    def create_new_object(self, name, entitytype, locations=None, description=''):
        assert locations is not None, f"Need to specify initial location for new entity <{name}:{entitytype}>"
        known_locs = set(locations)
        known_locs.discard(self._unknown_location)
        if known_locs:
            initial_loc = known_locs.pop()
            assert initial_loc is not None, f"Must include a known location: {locations}"
        else:
            initial_loc = self._unknown_location
            print(f"WARNING create_new_object({name},locations={locations}) with initial_loc=UNKNOWN!")
        if entitytype == DOOR:
            new_entity = Door(name=name, description=description, location=initial_loc)
        else:
            new_entity = Thing(name=name, description=description, location=initial_loc, type=entitytype)
        added_new = initial_loc.add_entity(new_entity)
        if len(locations) > 1:
            for loc in locations:
                loc.add_entity(new_entity)  # does nothing if already has_entity_with_name
            if entitytype != DOOR:
                print(f"WARNING: adding new {new_entity} to multiple locations: {locations}")
                assert False, "Shouldn't be adding non-door entity to multiple locations"
        # DISCARD NewEntityEvent -- self.gi.event_stream.push(ev)
        ev = None
        if added_new:  # this is a new entity with a known location
            ev = NewEntityEvent(new_entity)
            add_attributes_for_type(new_entity, entitytype)
            if not self.groundtruth and initial_loc != self._unknown_location:
                print("DISCOVERED NEW entity:", new_entity)
                self.event_stream.push(ev)
        return new_entity, ev

    def add_obj_to_obj(self, fact, rel=None):
        assert rel == fact.name
        # rel = fact.name
        o = fact.arguments[0]
        h = fact.arguments[1]
        if o.name.startswith('~') or h.name.startswith('~'):
            print("_add_obj_to_obj: SKIPPING FACT", fact)
            return None, None
        assert h.type != 'I'   # Inventory
        entity_set = self.entities_with_name(h.name, entitytype=entity_type_for_twvar(h.type))
        if entity_set:
            holder = entity_set.pop()  # choose one of them
            if len(entity_set):
                print("WARNING: found more than one matching entity for name <{}: {} + {}".format(
                    h.name, holder, entity_set))
            loc = self.location_of_entity(holder.name)
        else:
            print("WARNING: found no entities_with_name:", h.name)
            holder = None
            loc = None
        if not loc:
            print("WARNING! NO LOCATION FOR HOLDER while adding Object {} {}".format(fact, holder))
            print("unknown =", self._unknown_location)
            print("self._locations:", self._locations)
            loc_list = None
        else:
            loc_list = [loc]  # a list containing exactly one element

        #NOTE TODO: handle objects that have moved from to or from Inventory
        entitytype = entity_type_for_twvar(o.type)
        obj = self.get_entity(o.name, entitytype=entitytype)
        if not obj:
            obj, ev = self.create_new_object(o.name, entitytype, locations=loc_list)
            if ev:
                print("ADDED NEW Object {} :{}: {}".format(obj, fact.name, h.name))
                # if not self.groundtruth: gi.event_stream.push(ev)  # ALREADY DONE BY self.get_create_new_object()
        else:
            self.maybe_move_entity(obj, locations=loc_list)

        #add entity to entity (inventory is of type 'location', adding is done by create_if_notfound)
        if holder:
            holder.add_entity(obj, rel=rel)
        return obj, holder

    def add_obj_to_inventory(self, fact):
        # rel = fact.name
        o = fact.arguments[0]
        h = fact.arguments[1]
        if o.name.startswith('~'):
            assert False, f"_add_obj_to_inventory: UNEXPECTED FACT {fact}"
        assert h.name == 'I'   # Inventory
        loc_list = [self.inventory]  # a list containing exactly one location

        #NOTE TODO: handle objects that have moved from to or from Inventory
        entitytype = entity_type_for_twvar(o.type)
        obj = self.get_entity(o.name, entitytype=entitytype)
        if not obj:
            obj, ev = self.create_new_object(o.name, entitytype, locations=loc_list)
            if ev:
                print("ADDED NEW Object {} :in: Inventory".format(obj))
                # if not self.groundtruth:
                #     gi.event_stream.push(ev)  # ALREADY DONE BY self.get_entity()
        else:
            #print("FOUND GT Object {} :{}: {}".format(obj, fact.name, holder_for_logging))
            self.maybe_move_entity(obj, locations=loc_list)
        # DEBUGGING: redundant SANITY CHECK
        if not self.inventory.get_entity_by_name(obj.name):
            print(obj.name, "NOT IN INVENTORY", self.inventory.entities)
            assert False
        return obj

    def update_facts(self, obs_facts, prev_action=None):
        print(f"*********** ADDING FACTS (groundtruth={self.groundtruth})")
        player_loc = None
        door_facts = []
        at_facts = []
        on_facts = []
        in_facts = []
        inventory_facts = []
        other_facts = []
        for fact in obs_facts:
            a0 = fact.arguments[0]
            a1 = fact.arguments[1] if len(fact.arguments) > 1 else None
            if fact.name == 'link' and self.groundtruth:
                door_facts.append(fact)
            elif fact.name == 'at':
                if a0.type == 'P' and a1.type == 'r':
                    player_loc = self.get_location(a1.name, create_if_notfound=True)
                else:
                    at_facts.append(fact)
            elif fact.name == 'on':
                on_facts.append(fact)
            elif fact.name == 'in':
                if a1 and a1.type == 'I':  # Inventory
                    inventory_facts.append(fact)
                else:
                    in_facts.append(fact)
            elif fact.name in DIRECTION_ACTIONS:
                if a0.type == 'r' and a1.type == 'r' and self.groundtruth:
                    # During this initial pass we create locations and connections among them
                    # print('++CONNECTION:', fact)
                    loc0 = self.get_location(a0.name, create_if_notfound=True)
                    loc1 = self.get_location(a1.name, create_if_notfound=True)
                    # door_name = find_door(gt_facts, a1, a0)
                    # if door_name:
                    #     door = self._get_gt_entity(door_name, entitytype=DOOR, locations=[loc1, loc0], create_if_notfound=True)
                    # else:
                    #     door = None
                    door = None  # will add info about doors later
                    new_connection = Connection(loc1, DIRECTION_ACTIONS[fact.name], loc0, doorway=door)
                    self.add_connection(new_connection)  # does nothing if connection already present
                elif a0.type == 'd' and a1.type == 'r':
                    print("\t\tdoor fact -- ", fact)
                    door_facts.append(fact)
                    # pass
                else:
                    print("--IGNORING DIRECTION_FACT:", fact)
                    pass
            else:
                other_facts.append(fact)
        # 2nd pass, add doors to connections
        for fact in door_facts:
            r0 = None
            r1 = None
            d = None
            door_locations = []
            if len(fact.arguments) == 3:
                r0 = fact.arguments[0]
                d = fact.arguments[1]
                r1 = fact.arguments[2]
                assert fact.name == 'link'
                assert r0.type == 'r'
                assert r1.type == 'r'
                assert d.type == 'd'
                if not self.groundtruth:
                    print("WARNING (assertion failure) link fact in door_facts but not groundtruth:", fact)
                    continue   # link
            elif len(fact.arguments) == 2:
                r0 = fact.arguments[1]
                d = fact.arguments[0]
                assert fact.name in DIRECTION_ACTIONS
                assert r0.type == 'r'
                assert d.type == 'd'
                if self.groundtruth:
                    continue   # rely on 'link' facts to create doors
            if r0:
                loc0 = self.get_location(r0.name, create_if_notfound=False)
                door_locations.append(loc0)
            if r1:
                loc1 = self.get_location(r1.name, create_if_notfound=False)
                door_locations.append(loc1)
            else:
                door_locations.append(self._unknown_location)  # we don't yet know what's on the other side of the door

            # door, _ = self.get_entity(d.name,
            #                           entitytype=DOOR,
            #                           locations=door_locations,
            #                           create_if_notfound=True)
            door = self.get_entity(d.name, entitytype=DOOR)
            if not door:
                door, _ = self.create_new_object(d.name, DOOR, locations=door_locations)
            else:
                self.maybe_move_entity(door, locations=door_locations)
            if r1:  #len(door_locations) == 2:
                linkpath = self.connections.shortest_path(loc0, loc1)
                assert len(linkpath) == 1
                connection = linkpath[0]
                connection.doorway = door  # add DOOR to the Connection
            if self.groundtruth:
                door.state.open()   # assume that it's open, until we find a closed() fact...
        for fact in other_facts:
            if fact.name == 'closed' and fact.arguments[0].type == 'd':
                doorname = fact.arguments[0].name
                doors = self.entities_with_name(doorname, entitytype=DOOR)
                assert len(doors) == 1
                door = list(doors)[0]
                door.state.close()

        if player_loc:  # UPDATE player_location
            prev_loc = self.player_location
            if self.set_player_location(player_loc):
                if self.groundtruth:
                    print(f"GT knowledge graph updating player location from {prev_loc} to {player_loc}")
                else:
                    print(f"Action: <{prev_action}> CHANGED player location from {prev_loc} to {player_loc}")
                    if prev_action and player_loc != self._unknown_location and prev_loc != self._unknown_location:
                        verb = prev_action.split()[0]
                        direction_rel = verb + '_of'
                        if direction_rel in DIRECTION_ACTIONS:
                            door = None   # TODO: remember door directions relative to rooms, and look up the appropr door
                            new_connection = Connection(prev_loc,
                                                        DIRECTION_ACTIONS[direction_rel],
                                                        player_loc,
                                                        doorway=door)
                            self.add_connection(new_connection)
                # if player_loc != self._unknown_location:
        for fact in at_facts:
            # print("DEBUG at_fact", fact)
            o = fact.arguments[0]
            r = fact.arguments[1]
            loc = self.get_location(r.name, create_if_notfound=False)
            # print("DEBUG at_facts: LOCATION for roomname", r.name, "->", loc)
            if loc:
                locs = [loc]
            else:
                print("WARNING: at_facts failed to find location: ", r.name)
                print(self.player_location, self.locations)
                locs = None
            if r.type == 'r':
                entitytype = entity_type_for_twvar(o.type)
                obj = self.get_entity(o.name, entitytype=entitytype)
                if not obj:
                    obj, _ = self.create_new_object(o.name, entitytype, locations=locs)
                else:
                    self.maybe_move_entity(obj, locations=locs)
            else:
                gv.dbg("WARNING -- ADD FACTS: unexpected location for at(o,l): {}".format(r))
            # add_attributes_for_type(obj, o.type)

        #NOTE: the following assumes that objects are not stacked on top of other objects which are on or in objects
        # and similarly, that "in" relations are not nested.
        #TODO: this should be generalized to work correctly for arbitrary chains of 'on' and 'in'
        #(currently assumes only one level: container or holder is immobile, previously identified "at" a place)

        for fact in on_facts:
            # print("DEBUG on_fact", fact)
            o1, o2 = self.add_obj_to_obj(fact, rel='on')
            if o1 and o2:
                add_attributes_for_predicate(o1, 'on', o2)
        for fact in in_facts:
            o1, o2 = self.add_obj_to_obj(fact, rel='in')
            if o1 and o2:
                add_attributes_for_predicate(o1, 'in', o2)
        for fact in inventory_facts:
            o1 = self.add_obj_to_inventory(fact)

        for fact in other_facts:
            predicate = fact.name
            if predicate == 'cooking_location' \
            or predicate.startswith('ingredient_') \
            or predicate == 'free' \
            or predicate == 'base' \
            or predicate == 'out':
                continue

            a0 = fact.arguments[0]
            o1 = None
            o2 = None
            if a0.name.startswith('~'):
                continue
            if len(fact.arguments) > 1:
                a1 = fact.arguments[1]
            else:
                a1 = None
            o1 = self.get_entity(a0.name, entitytype=entity_type_for_twvar(a0.type))
            if a1:
                o2 = self.get_entity(a1.name, entitytype=entity_type_for_twvar(a1.type))
            if o1:
                add_attributes_for_predicate(o1, predicate, entity2=o2)
            else:
                if predicate == 'edible' and a0.name == 'meal':
                    continue
                print("Warning: add_attributes_for_predicate", predicate, "didnt find an entity corresponding to", a0)


class ConnectionGraph:
    """
    Graph of connections between locations.

    """
    def __init__(self):
        self._out_graph = {} # Location : [Outgoing Connections]
        self._in_graph  = {} # Location : [Incoming Connections]

    def add(self, connection, kg: KnowledgeGraph):
        """ Adds a new connection to the graph if it doesn't already exist. """
        from_location = connection.from_location
        to_location = connection.to_location
        kg.event_stream.push(NewConnectionEvent(connection, groundtruth=kg.groundtruth))
        if from_location in self._out_graph:
            if connection in self._out_graph[from_location]:
                # print("IGNORING add(connection:", connection)
                return
            self._out_graph[from_location].append(connection)
        else:
            self._out_graph[from_location] = [connection]
        if to_location is not None:
            if to_location in self._in_graph:
                self._in_graph[to_location].append(connection)
            else:
                self._in_graph[to_location] = [connection]
        if not kg.groundtruth:
            print("ADDED NEW {}CONNECTION".format('GT ' if kg.groundtruth else ''), connection)

    def incoming(self, location):
        """ Returns a list of incoming connections to the given location. """
        if location in self._in_graph:
            return self._in_graph[location]
        else:
            return []

    def outgoing(self, location):
        """ Returns a list of outgoing connections from the given location. """
        if location in self._out_graph:
            return self._out_graph[location]
        else:
            return []

    def navigate(self, location, nav_action):
        """Returns the destination that is reached by performing nav_action
        from location.
        """
        if not isinstance(nav_action, Action):
            raise ValueError("Expected Action. Got {}".format(type(nav_action)))
        for connection in self.outgoing(location):
            if connection.action == nav_action:
                return connection.to_location
        return None

    def shortest_path(self, start_location, end_location, path=[]):
        """ Find the shortest path between start and end locations. """
        if start_location == end_location:
            return path
        if start_location not in self._out_graph:
            return None
        shortest = None
        for connection in self._out_graph[start_location]:
            if connection not in path:
                newpath = self.shortest_path(connection.to_location,
                                             end_location,
                                             path + [connection])
                if newpath:
                    if not shortest or len(newpath) < len(shortest):
                        shortest = newpath
        return shortest


def _format_doorinfo(doorentity):
    if doorentity is None:
        return ''
    if doorentity.state.openable and not doorentity.state.is_open:
        return ":{}(closed)".format(doorentity.name)
    return ":{}(open)".format(doorentity.name)

class Connection:
    """
    A connection between two locations:

    from_location: The location that was departed
    action: The navigational action used
    to_location: The location arrived at, or None
    message: The text response given by the game upon moving

    """
    def __init__(self, from_location, action, to_location=None, doorway=None, message=''):
        self.from_location = from_location
        self.to_location   = to_location
        self.action        = action
        self.message       = message
        self.doorway       = doorway

    def __eq__(self, other):
        if isinstance(self, other.__class__):
            return self.action == other.action and\
                self.from_location == other.from_location and\
                self.to_location == other.to_location
        return False

    def to_string(self, prefix=''):
        return prefix + "{} --({}{})--> {}".format(self.from_location.name,
                                                   self.action,
                                                   _format_doorinfo(self.doorway),
                                                   self.to_location.name)

    def __str__(self):
        return self.to_string()

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return hash(self.text())
