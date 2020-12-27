import itertools
from collections import deque

def count_iter_items(iterable):
    """
    Consume an iterable not reading it into memory; return the number of items.
    """
    counter = itertools.count()
    deque(zip(iterable, counter), maxlen=0)  # (consume at C speed)
    return next(counter)


def reformat_go_skill(sk):
    if sk.startswith("go"):
        n = int(sk[2:])
        return f"go{n:02d}"
    else:
        return sk


def split_gamename(gname):
    pieces = gname.split('-')
    if len(pieces) >= 4 and pieces[0] == 'tw' and pieces[1] == 'cooking':
        skills = pieces[2]
        gid = pieces[3]
    else:
        skills = pieces[0]
        gid = pieces[1]
    sklist = skills.split('+')
    return gid, [ reformat_go_skill(sk) for sk in sklist ]

# def split_gamename(gname):
#     skills, gid = gname.split('-')
#     sklist = skills.split('+')
#     return gid, [reformat_go_skill(sk) for sk in sklist]


def parse_gameid(game_name: str) -> str:
    game_id = game_name[11:] if game_name.startswith("tw-cooking-") else game_name

    segments = game_id.split('-')
    if len(segments) >= 2:
        code, guid = segments[0:2]
        guid = guid.split('.')[0]
        guid = "{}..{}".format(guid[0:4], guid[-4:])
        segments = code.split('+')
        r, t, g, k, c, o, d = ('0', '0', 0, '_', '_', '_', '_')
        for seg in segments:
            if seg.startswith('recipe'):
                r = seg[len('recipe'):]
            elif seg.startswith('go'):
                g = int(seg[len('go'):])
            elif seg.startswith('take'):
                t = seg[len('take'):]
            elif seg == 'cook':
                k = 'k'
            elif seg == 'cut':
                c = 'c'
            elif seg == 'open':
                o = 'o'
            elif seg == 'drop':
                d = 'd'
            else:
                assert False, "unparsable game_id: {}".format(game_id)
        shortcode = "r{}t{}{}{}{}{}g{:02d}-{}".format(r, t, k, c, o, d, g, guid)
    else:
        shortcode = game_id
    return shortcode
