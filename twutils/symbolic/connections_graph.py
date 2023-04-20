from .entity import ConnectionRelation, Location
from .action import Action, GoNorth, GoSouth, GoEast, GoWest
from twutils.twlogic import reverse_direction

DIRECTION_ACTIONS = {
        'north_of': GoNorth,
        'south_of': GoSouth,
        'east_of': GoEast,
        'west_of': GoWest}

def map_action_to_direction(action):
    for key, val in DIRECTION_ACTIONS.items():
        if val == action:
            return key
    return None

def simple_direction(direction_rel:str) -> str:
    """
      Extract a direction (e.g. 'west') given a direction relation (e.g. 'west_of').
      If direction_rel is already a basic direction, returns it unmodified.
    """
    if direction_rel.endswith('_of'):
        return direction_rel[:-3]
    return direction_rel


class ConnectionGraph:
    """
    Graph of connections between locations.

    """
    def __init__(self, logger=None):
        self._out_graph = {}  # {from_Location : {direction: Connection(outgoing)} }
        self._in_graph  = {}  # {to_Location : { tuple(from_Location, direction): Connection(incoming)} }
        self._logger = logger

    def set_logger(self, logger):
        self._logger = logger

    def dbg(self, msg):
        if self._logger:
            logger = self._logger
            logger.debug(msg)
        else:
            print("### DEBUG:", msg)

    def warn(self, msg):
        if self._logger:
            logger = self._logger
            logger.warning(msg)
        else:
            print("### WARNING:", msg)

    def add_connection(self, from_location, direction_rel, to_location=None, door=None, assume_inverse=False):
        if direction_rel not in DIRECTION_ACTIONS:
            self.warn(f"WARNING: UNEXPECTED direction_rel: '{direction_rel}'")
        else:
            new_connection = Connection(from_location, DIRECTION_ACTIONS[direction_rel], to_location=to_location, doorway=door)
            self._add(new_connection, assume_inverse=assume_inverse)

    def _add(self, connection, assume_inverse=False):
        """ Adds a new connection to the graph if it doesn't already exist,
            or updates an existing connection (if to_location was previously UnknownLocation)
            Does nothing if a similar connection already exists
         """
        from_location = connection.from_location
        assert not Location.is_unknown(from_location)
        to_location = connection.to_location
        direction = map_action_to_direction(connection.action)
        added_new = []
        # kg.broadcast_event(NewConnectionEvent(connection, groundtruth=kg.is_groundtruth))
        if from_location not in self._out_graph:
            added_new.append(f"out_graph[{from_location.name}]")
            self._out_graph[from_location] = {direction: connection}
        else:
            if direction in self._out_graph[from_location]:
                # if self._out_graph[from_location][direction] != connection:
                    # print(f"... updating {self._out_graph[from_location][direction]} <= {connection}")
                connection = self._out_graph[from_location][direction].update(connection)
            else:
                added_new.append(f"out_graph[{from_location.name}][{direction}]")
                self._out_graph[from_location][direction] = connection

        if not Location.is_unknown(to_location):   # don't index connections incoming to UnknownLocation
            incoming_rel = ConnectionRelation(from_location=from_location, direction=direction)
            if to_location not in self._in_graph:
                added_new.append(f"in_graph[{to_location.name}] {incoming_rel}")
                self._in_graph[to_location] = {incoming_rel: connection}
            else:
                if incoming_rel in self._in_graph[to_location]:
                    # if self._in_graph[to_location][incoming_rel] != connection:
                    #     print(f"... updating {self._in_graph[to_location][incoming_rel]} <= {connection}")
                    self._in_graph[to_location][incoming_rel].update(connection)
                else:
                    added_new.append(f"in_graph[{to_location.name}][{incoming_rel}]")
                    self._in_graph[to_location][incoming_rel] = connection

        # if there's a doorway associated with this connection,
        # we might be able to use its 2nd endpoint to update or create a reverse connection
        # (going back the other way through the same door)
        if connection.doorway and \
                connection.doorway.direction_from_loc1 and \
                connection.doorway.direction_from_loc2 and \
                not Location.is_unknown(connection.doorway.direction_from_loc2.from_location):
            assert not Location.is_unknown(connection.doorway.direction_from_loc1.from_location)
            rev_direction = None
            if connection.doorway.direction_from_loc1.from_location == connection.from_location:
                if connection.doorway.direction_from_loc1.direction:
                    assert connection.doorway.direction_from_loc1.direction == direction
                to_location = connection.from_location
                from_location = connection.doorway.direction_from_loc2.from_location
                rev_direction = connection.doorway.direction_from_loc2.direction
            else:
                assert connection.doorway.direction_from_loc2.from_location == connection.from_location
                if connection.doorway.direction_from_loc2.direction:
                    assert connection.doorway.direction_from_loc2.direction == direction
                if Location.is_unknown(connection.to_location):  # maybe fill it in using doorway info
                    to_location = connection.from_location
                    from_location = connection.doorway.direction_from_loc1.from_location
                    rev_direction = connection.doorway.direction_from_loc1.direction
            if rev_direction:
                self.add_connection(
                    from_location,
                    rev_direction,
                    to_location=to_location,
                    door=connection.doorway,
                    assume_inverse=False)
                assume_inverse = False   # since we've just built a reverse connection, skip default inverse logic

        # if added_new and not kg.is_groundtruth:
        #         print("\tADDED NEW {}CONNECTION".format('GT ' if kg.is_groundtruth else ''), added_new, connection)

        if assume_inverse:  # assume that 180 inverse direction connects to_location => from_location
            if not Location.is_unknown(connection.to_location):
                self.add_connection(
                    connection.to_location,
                    reverse_direction(direction),
                    to_location=from_location,
                    door=connection.doorway,
                    assume_inverse=False)

    def incoming(self, location):
        """ Returns a list of incoming connections to the given location. """
        if location in self._in_graph:
            return list(self._in_graph[location].values())
        else:
            return []

    def outgoing(self, location):
        """ Returns a list of outgoing connections from the given location. """
        if location in self._out_graph:
            return list(self._out_graph[location].values())
        else:
            return []

    def outgoing_directions(self, location):
        if location not in self._out_graph:
            return []
        return list(map(simple_direction, self._out_graph[location].keys()))

    def connection_for_direction(self, location, direction):   # returns None if no exit that direction
        if not direction.endswith('_of'):
            direction = direction+'_of'    # convert e.g. north => north_of
        return self._out_graph[location].get(direction, None)

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

    def shortest_path(self, start_location, end_location, bestpath=None, visited=None):
        """ Find the shortest path between start and end locations. """
        if bestpath is None:
            bestpath = []
        if visited is None:
            visited = []
        if start_location == end_location:
            return bestpath
        if start_location not in self._out_graph:
            return None
        if start_location in visited:
            # print(f"shortest_path: SKIPPING {start_location} to avoid looping")
            return None
        visited.append(start_location)
        shortest = None
        for connection in self.outgoing(start_location):
            if connection not in bestpath:
                newpath = self.shortest_path(connection.to_location,
                                             end_location,
                                             bestpath + [connection], visited=visited)
                if newpath:
                    if not shortest or len(newpath) < len(shortest):
                        shortest = newpath
        return shortest

    def to_string(self):
        strout = "Connection Graph:\n\n"
        strout += "    outgoing:\n"
        for loc in self._out_graph:
            for con in self.outgoing(loc):
                strout += con.to_string()+'\n'
        strout += "\n    incoming:\n"
        for loc in self._in_graph:
            for con in self.incoming(loc):
                strout += con.to_string()+'\n'
        return strout

    def __str__(self):
        return self.to_string()


def _format_doorinfo(doorentity, options=None):
    if doorentity is None:
        return ''
    if options and options.startswith('kg-descr') or options == 'parsed-obs':
        if doorentity.state.openable and not doorentity.state.is_open:
            return f" +closed {doorentity.name}"
        return f" +open {doorentity.name}"
    #else:
    if doorentity.state.openable and not doorentity.state.is_open:
        return f"{doorentity.name}(closed)"
    return f"{doorentity.name}(open)"


def _format_location(location, options=None):
    if not location:
        return ''
    if options and options.startswith('kg-descr') or options == 'parsed-obs':
        return "unknown" if Location.is_unknown(location) else location.name

    return ":{}[{}]".format('', location.name)


class Connection:
    """
    A connection between two locations:

    from_location: The location that was departed
    action: The navigational action used
    to_location: The location arrived at, or None
    message: The text response given by the game upon moving

    """
    def __init__(self, from_location, action, to_location=None, doorway=None):
        self.from_location = from_location
        self.to_location   = to_location
        self.action        = action
        self.doorway       = doorway

    def __eq__(self, other):
        if isinstance(self, other.__class__):
            return self.action == other.action and\
                self.from_location == other.from_location and\
                self.to_location == other.to_location and\
                self.action == other.action and \
                self.doorway == other.doorway
        return False

    def update(self, other: 'Connection') -> 'Connection':
        updated = []
        if not Location.is_unknown(other.from_location):
            if self.from_location != other.from_location:
                assert Location.is_unknown(self.from_location), \
                    f"Connection endpoints aren't expected to change over time {self} <= {other}"
                updated.append("from_location")
                self.from_location = other.from_location
        if not Location.is_unknown(other.to_location):
            if self.to_location != other.to_location:
                assert Location.is_unknown(self.to_location), \
                    f"Connection destinations aren't expected to change over time {self} <= {other}"
                updated.append("to_location")
                self.to_location = other.to_location
        if other.action:
            if self.action != other.action:
                print(f"WARNING: replacing connection action {self} with {other.action}")
                updated.append("action")
            self.action = other.action
        if other.doorway:
            if self.doorway:
                assert self.doorway == other.doorway
            else:
                updated.append("doorway")
            self.doorway = other.doorway
        # if updated:
        #     print(f"ConnectionGraph updated {self} {updated} from {other}")
        return self

    def to_string(self, prefix='', options=None):
        if options and options.startswith('kg-descr'):       # info known by current (non-GT) knowledge graph
            return prefix + "{}{} to {}".format(self.action.verb,
                    _format_doorinfo(self.doorway, options=options),
                    _format_location(self.to_location, options=options))
        elif options == 'parsed-obs':   # info directly discernible from 'look' command
            return prefix + "{}{}".format(self.action.verb,
                    _format_doorinfo(self.doorway, options=options))
        elif options:
            assert False, f"Unknown formatting options: {options}"
        return prefix + "{} --({}:{})--> {}".format(_format_location(self.from_location),
                                                   self.action.verb,
                                                   _format_doorinfo(self.doorway),
                                                   _format_location(self.to_location))

    def __str__(self):
        return self.to_string()

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return hash(self.to_string())
