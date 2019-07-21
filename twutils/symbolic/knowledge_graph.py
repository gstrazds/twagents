# import os, sys
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from symbolic.event import *
from symbolic.action import Action
from symbolic.location import Location, Inventory
from fuzzywuzzy import fuzz


class KnowledgeGraph:
    """
    Knowledge Representation consists of visisted locations.

    """
    def __init__(self, groundtruth=False):
        self._locations          = []
        self._player_location    = None
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

    def most_similar_location(self, description):
        """ Returns the location with the highest similarity to the given description. """
        possible_name = Location.extract_name(description)
        existing_locs = self.locations_with_name(possible_name)
        if not existing_locs:
            existing_locs = self._locations
        most_similar = None
        best_similarity = 0
        for loc in existing_locs:
            similarity = fuzz.partial_ratio(loc.description, description)
            if similarity > best_similarity:
                best_similarity = similarity
                most_similar = loc
        return most_similar

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
        self.inventory.reset()
        for location in self.locations:
            location.reset()

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
        print("ADDED NEW {}CONNECTION".format('GT ' if groundtruth else ''), connection)

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



class Connection:
    """
    A connection between two locations:

    from_location: The location that was departed
    action: The navigational action used
    to_location: The location arrived at, or None
    message: The text response given by the game upon moving

    """
    def __init__(self, from_location, action, to_location=None, message=''):
        self.from_location = from_location
        self.to_location   = to_location
        self.action        = action
        self.message       = message

    def __eq__(self, other):
        if isinstance(self, other.__class__):
            return self.action == other.action and\
                self.from_location == other.from_location and\
                self.to_location == other.to_location
        return False

    def to_string(self, prefix=''):
        return prefix + "{} --({})--> {}".format(self.from_location.name,
                                                 self.action,
                                                 self.to_location.name)

    def __str__(self):
        return self.to_string()

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return hash(self.text())
