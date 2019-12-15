# import os, sys
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from symbolic.event import *
from symbolic.action import *
from symbolic.entity import Entity
from symbolic.location import Location, Inventory

DIRECTION_ACTIONS = {
        'north_of': GoNorth,
        'south_of': GoSouth,
        'east_of': GoEast,
        'west_of': GoWest}

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
    if twvartype == 'd':  # if a DOOR
        if not entity.state.openable:
            entity.state.open()  # assume default until we determine otherwise (via add_attributes_for_predicate)


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


class KnowledgeGraph:
    """
    Knowledge Representation consists of visisted locations.

    """
    def __init__(self, groundtruth=False):
        self._locations          = []
        self._player_location    = Location("<UNKNOWN>")
        self._init_loc           = None
        self._inventory          = Inventory()
        self._connections        = ConnectionGraph()
        self.groundtruth         = groundtruth

    @property
    def locations(self):
        return self._locations

    def add_location(self, new_location: Location) -> NewLocationEvent:
        """ Adds a new location object and broadcasts a NewLocation event. """
        self._locations.append(new_location)
        return NewLocationEvent(new_location, groundtruth=self.groundtruth)

    def locations_with_name(self, location_name):
        """ Returns all locations with a particular name. """
        return [l for l in self._locations if l.name == location_name]

    @property
    def player_location(self):
        return self._player_location

    # @player_location.setter
    def set_player_location(self, new_location, gi):
        """ Changes player location and broadcasts a LocationChangedEvent. """
        if new_location == self._player_location:
            return False
        if new_location:
            gi.event_stream.push(LocationChangedEvent(new_location, groundtruth=self.groundtruth))
        self._player_location = new_location
        return True

    @property
    def inventory(self):
        return self._inventory

    @property
    def connections(self):
        return self._connections

    def add_connection(self, new_connection, gi):
        """ Adds a connection object. """
        self._connections.add(new_connection, gi, groundtruth=self.groundtruth)

    def reset(self, gi):
        """Returns the knowledge_graph to a state resembling the start of the
        game. Note this does not remove discovered objects or locations. """
        self.set_player_location(self._init_loc, gi)
        self.inventory.reset(gi)
        for location in self.locations:
            location.reset(gi)

    def __str__(self):
        s = "Knowledge Graph{}\n".format('[GT]' if self.groundtruth else '')
        if self._player_location == None:
            s += "PlayerLocation: None"
        else:
            s += "PlayerLocation: {}".format(self._player_location.name)
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
        for l in (self._locations + [self._inventory]):
            e = l.get_entity_by_name(entityname)
            if e and (entitytype is None or e._type == entitytype):
                ret.add(e)
        # if len(ret) == 0:   # check also on or in other entities (just one level deep)
        #     for l in (self._locations + [self._inventory]):
        #         for e in l.entities:
        #             for e2 in e._entities:
        #                 if e2 and e2.has_name(entityname) and (entitytype is None or e2._type == entitytype):
        #                     ret.add(e2)
        return ret

    def location_of_entity(self, entityname, entitytype=None):
        """ Returns locations where an entity with a specific name can be found """
        ret = set()
        for l in (self._locations + [self._inventory]):
            e = l.get_entity_by_name(entityname)
            if e and (entitytype is None or e._type == entitytype):
                ret.add(l)
        # if len(ret) == 0:   # check also on or in other entities (just one level deep)
        #     for l in (self._locations + [self._inventory]):
        #         for e in l.entities:
        #             for e2 in e._entities:
        #                 if e2 and e2.has_name(entityname) and (entitytype is None or e2._type == entitytype):
        #                     ret.add(l)
        return ret

    def get_containing_entity(self, entity):
        """ Returns container (or support) where an entity with a specific name can be found """
        for l in self._locations:
            for ce in l.entities:
                if entity in ce._entities:
                    return ce
        # if len(ret) == 0:   # check also on or in other entities (just one level deep)
        #     for l in (self._locations + [self._inventory]):
        #         for e in l.entities:
        #             for e2 in e._entities:
        #                 if e2 and e2.has_name(entityname) and (entitytype is None or e2._type == entitytype):
        #                     ret.add(l)
        return None

    def get_location(self, roomname, gi, create_if_notfound=False):
        locations = self.locations_with_name(roomname)
        if locations:
            assert len(locations) == 1
            return locations[0]
        elif create_if_notfound:
            new_loc = Location(roomname)
            ev = self.add_location(new_loc)
            if ev and not self.groundtruth:
                gi.event_stream.push(ev)
            # if self.groundtruth: DISCARD NewlocationEvent else gi.event_stream.push(ev)
#            print("created new GT Location:", new_loc)
            return new_loc
        return None

    def get_entity(self, name, gi, locations=None, entitytype=None, create_if_notfound=False):
        if create_if_notfound:
            assert locations is not None, f"Need to specify initial location for new entity <{name}:{entitytype}>"
        entities = set()
        if locations:
            for l in locations:
                e = l.get_entity_by_name(name)
                if e:
                    entities.add(e)
        if not entities:  # none found
            entities = self.entities_with_name(name, entitytype=entitytype)
            if locations and entities:
                # need to move it from wherever it was to its new location
                print(f"get_entity() WARNING - MOVING {entities} to {locations}")
                prev_loc = self.location_of_entity(name, entitytype=entitytype)
                if prev_loc and len(prev_loc) == 1:
                    l = prev_loc.pop()
                    e = list(entities)[0]
                    # TODO: here we are assuming exactly one entity and one location
                    l.del_entity(e)
                    locations[0].add_entity(e)
                    # TODO: send an EntityMovedEvent
                    if not self.groundtruth:
                        # gi.event_stream.push(EntityMovedEvent(e, l, locations[0], groundtruth=self.groundtruth))
                        pass
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

            added_new = locations[0].add_entity(new_entity)
            if len(locations) > 0:
                for l in locations[1:]:
                    l.add_entity(new_entity)
                    if entitytype != 'd':
                        print(f"WARNING: adding new {new_entity} to multiple locations: {locations}")
                        assert False, "Shouldn't be adding non-door entity to multiple locations"
            # DISCARD NewEntityEvent -- self.gi.event_stream.push(ev)
            ev = None
            if added_new:  # this is a new entity
                ev = NewEntityEvent(new_entity)
                add_attributes_for_type(new_entity, entitytype)
                if not self.groundtruth:
                    print("created new:", new_entity)
                    gi.event_stream.push(ev)
            return new_entity, ev
        return None, None

    def add_obj_to_obj(self, gi, fact, player_loc):
        o = fact.arguments[0]
        h = fact.arguments[1]
        if o.name.startswith('~') or h.name.startswith('~'):
            print("_add_obj_to_obj: SKIPPING FACT", fact)
            return None, None
        if h.name == 'I':  # Inventory
            holder = None  #self.gi.gt.inventory
            loc = [ self.inventory ] #player_loc
        else:
            # holder, _ = self._get_gt_entity(h.name, entitytype=entity_type_for_twvar(h.type), locations=None,
            #                              create_if_notfound=False)
            entity_set = self.entities_with_name(h.name, entitytype=entity_type_for_twvar(h.type))
            if entity_set:
                holder = entity_set.pop()  # choose one of them
                if len(entity_set):
                    print("WARNING: found more than one matching entity for name <{}: {} + {}".format(
                        h.name, holder, entity_set))
            else:
                holder = None
                loc = None
            if holder:
                # loc = holder._init_loc if holder is not None else None
                locs = self.location_of_entity(holder.name)
                if locs:
                    loc = list(locs)  # we want a list of locations (ideally containing only one element)
                    if len(locs) > 1:
                        print("WARNING: entity <{}> has multiple locations: {}".format(holder, loc))
                        loc = list(loc[0])  # if multiple locations, choose just one
        if not loc:
            print("WARNING! NO LOCATION FOR HOLDER while adding Object {} {}".format(fact, holder))
            loc = None

    #NOTE TODO: handle objects that have moved from to or from Inventory
        obj, ev = self.get_entity(o.name, gi, entitytype=entity_type_for_twvar(o.type), locations=loc,
                                  create_if_notfound=True)
        #add entity to entity (inventory is of type 'location', adding is done by create_if_notfound)
        if holder:
            holder.add_entity(obj)

        holder_for_logging = 'Inventory' if h.name == 'I' else holder
        if ev:
            print("ADDED NEW Object {} :{}: {}".format(obj, fact.name, holder_for_logging))
            # if not self.groundtruth:
            #     gi.event_stream.push(ev)  # ALREADY DONE BY self.get_entity()
        else:
            #print("FOUND GT Object {} :{}: {}".format(obj, fact.name, holder_for_logging))
            if holder_for_logging == 'Inventory':
                if not self.inventory.get_entity_by_name(obj.name):
                    print(obj.name, "NOT IN INVENTORY", self.inventory.entities)
                    assert False
        return obj, holder

    def add_facts(self, obs_facts, gi):
        print(f"ADDING FACTS (groundtruth={self.groundtruth})")
        player_loc = None
        door_facts = []
        at_facts = []
        on_facts = []
        in_facts = []
        other_facts = []
        for fact in obs_facts:
            a0 = fact.arguments[0]
            a1 = fact.arguments[1] if len(fact.arguments) > 1 else None
            if fact.name == 'link':
                door_facts.append(fact)
            elif fact.name == 'at':
                at_facts.append(fact)
                if a0.type == 'P' and a1.type == 'r':
                    player_loc = self.get_location(a1.name, gi, create_if_notfound=True)
            elif fact.name == 'on':
                on_facts.append(fact)
            elif fact.name == 'in':
                in_facts.append(fact)
            elif fact.name in DIRECTION_ACTIONS:
                if a0.type == 'r' and a1.type == 'r':
                    # During this initial pass we create locations and connections among them
                    # print('++CONNECTION:', fact)
                    loc0 = self.get_location(a0.name, gi, create_if_notfound=True)
                    loc1 = self.get_location(a1.name, gi, create_if_notfound=True)
                    # door_name = find_door(gt_facts, a1, a0)
                    # if door_name:
                    #     door = self._get_gt_entity(door_name, entitytype=gv.DOOR, locations=[loc1, loc0], create_if_notfound=True)
                    # else:
                    #     door = None
                    door = None  # will add info about doors later
                    new_connection = Connection(loc1, DIRECTION_ACTIONS[fact.name], loc0, doorway=door)
                    self.add_connection(new_connection, gi)  # does nothing if connection already present
                elif a0.type == 'd' and a1.type == 'r':
                    print("\t\tTODO door fact -- ", fact)
                    door_facts.append(fact)
                    # pass
                else:
                    # print("--IGNORING:", fact)
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
                assert r0.type == 'r'
                assert r1.type == 'r'
                assert d.type == 'd'
            elif len(fact.arguments) == 2:
                r0 = fact.arguments[1]
                d = fact.arguments[0]
                assert r0.type == 'r'
                assert d.type == 'd'

            if r0:
                loc0 = self.get_location(r0.name, gi, create_if_notfound=False)
                door_locations.append(loc0)
            if r1:
                loc1 = self.get_location(r1.name, gi, create_if_notfound=False)
                door_locations.append(loc1)
            door, _ = self.get_entity(d.name, gi, entitytype=gv.DOOR,
                                            locations=door_locations, create_if_notfound=True)
            if len(door_locations) == 2:
                linkpath = self.connections.shortest_path(loc0, loc1)
                assert len(linkpath) == 1
                connection = linkpath[0]
                connection.doorway = door
            door.state.open()   # assume that it's open, until we find a closed() fact...
        for fact in other_facts:
            if fact.name == 'closed' and fact.arguments[0].type == 'd':
                doorname = fact.arguments[0].name
                doors = self.entities_with_name(doorname, entitytype=gv.DOOR)
                assert len(doors) == 1
                door = list(doors)[0]
                door.state.close()
        for fact in at_facts:
            o = fact.arguments[0]
            r = fact.arguments[1]
            loc = self.get_location(r.name, gi, create_if_notfound=False)
            if loc:
                locs = [loc]
            else:
                print("WARNING: failed to find location: ", r.name)
                print(self.player_location, self.locations)
                locs = None
            if r.type == 'r':
                obj, _ = self.get_entity(o.name, gi, entitytype=entity_type_for_twvar(o.type),
                                               locations=locs, create_if_notfound=True)
            else:
                gv.dbg("WARNING -- SET GROUND TRUTH: unexpected location for at(o,l): {}".format(r))
            # add_attributes_for_type(obj, o.type)

    #NOTE: the following assumes that objects are not stacked on top of other objects which are on or in objects
    # and similarly, that "in" relations are not nested.
    #TODO: this should be generalized to work correctly for arbitrary chains of 'on' and 'in'
    #(currently assumes only one level: container or holder is immobile, previously identified "at" a place)
        for fact in on_facts:
            o1, o2 = self.add_obj_to_obj(gi, fact, player_loc)
            if o1 and o2:
                add_attributes_for_predicate(o1, 'on', o2)
        for fact in in_facts:
            o1, o2 = self.add_obj_to_obj(gi, fact, player_loc)
            if o1 and o2:
                add_attributes_for_predicate(o1, 'in', o2)
        if player_loc:
            if self.set_player_location(player_loc, gi):
                print("CHANGED player location:", player_loc)

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
            o1, _ = self.get_entity(a0.name, gi, entitytype=entity_type_for_twvar(a0.type))
            if a1:
                o2, _ = self.get_entity(a1.name, gi, entitytype=entity_type_for_twvar(a1.type))
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

    def add(self, connection, gi, groundtruth=False):
        """ Adds a new connection to the graph if it doesn't already exist. """
        from_location = connection.from_location
        to_location = connection.to_location
        gi.event_stream.push(NewConnectionEvent(connection, groundtruth=groundtruth))
        if from_location in self._out_graph:
            if connection in self._out_graph[from_location]:
                # print("IGNORING new_connection:", connection)
                return
            self._out_graph[from_location].append(connection)
        else:
            self._out_graph[from_location] = [connection]
        if to_location is not None:
            if to_location in self._in_graph:
                self._in_graph[to_location].append(connection)
            else:
                self._in_graph[to_location] = [connection]
        #print("ADDED NEW {}CONNECTION".format('GT ' if groundtruth else ''), connection)

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
