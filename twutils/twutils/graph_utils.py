from collections import defaultdict, namedtuple, OrderedDict
import networkx as nx
from symbolic.entity import get_doorinfo

PointTuple = namedtuple("PointTuple", "x y")


class Point(PointTuple):
    def as_tuple(self):
        return (self.x, self.y)


DELTA_COORDS = {
    'east': Point(1, 0),
    'west': Point(-1, 0),
    'north': Point(0, 1),
    'south': Point(0, -1)
}

DELTA_COORDS_REV = {
    'west': Point(1, 0),
    'east': Point(-1, 0),
    'south': Point(0, 1),
    'north': Point(0, -1)
}

#def delta_coords(direction):
#    return

def room_graph_from_kg(kg, reverse_dir=False):
    cg = kg.connections

    room_graph = nx.Graph()

    _room_dict = {}

    #     def _already_visited(name):
    #         if name in _room_dict and _room_dict[name] is not None:
    #             return _room_dict[name]
    #         else:
    #             return None

    def visit_room(name, coords, from_node=None):
        prev_coords = _room_dict.get(name, None)
        # print("visit_room", name, coords, prev_coords)
        if prev_coords is not None:
            if coords is not None:
                if coords != prev_coords:
                    assert name.startswith('Unkn')
                    return visit_room(f"Unkn_{coords.x}_{coords.y}", coords, from_node=from_node)
        if coords is not None:
            if prev_coords is None:
                _room_dict[name] = coords
                label_for_graph = "Unknown" if name.startswith("Unkn") else name
                room_graph.add_node(coords.as_tuple(), name=label_for_graph)
                if from_node is not None:
                    room_graph.add_edge(from_node.as_tuple(), coords.as_tuple())
                    room_graph[from_node.as_tuple()][coords.as_tuple()]['has_door'] = False
                #             print("ADDED NODE:", coords, name)
                return True
        return False  # no change

    current_coords = Point(0, 0)
    outgoing_connections = list(cg._out_graph)
    if len(outgoing_connections):
        visit_room(outgoing_connections[0].name, current_coords)
    else:
        visit_room(kg.player_location.name, current_coords)

    for loc in outgoing_connections:
        visit_room(loc.name, None)

    finished = False
    while not finished:
        #         print()
        finished = True
        for loc in outgoing_connections:
            current_coords = _room_dict.get(loc.name, None)
            if current_coords is not None:
                for con in cg.outgoing(loc):
                    doorname, is_open = get_doorinfo(con.doorway)
                    if reverse_dir:
                        delta = DELTA_COORDS_REV[con.action.verb]
                    else:
                        delta = DELTA_COORDS[con.action.verb]

                    new_coords = Point(current_coords.x + delta.x, current_coords.y + delta.y)
                    if visit_room(con.to_location.name, new_coords, from_node=current_coords):
                        finished = False
                        if doorname:
                            e = room_graph[current_coords.as_tuple()][new_coords.as_tuple()]
                            e['has_door'] = True
                            e['door_state'] = 'open' if is_open else 'closed'
                            e['door_name'] = doorname
    #                     door_descr = f"<{'open' if is_open else 'closed'}>{doorname}" if doorname else ''
    #                     print("{} {} :{}: {} {}".format("**" if not finished else "  ",
    #                                                         con.from_location,
    #                                                         con.action.verb, door_descr,
    #                                                         con.to_location))
    return room_graph

