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


def split_gamename(gname:str):  # -> (str, str):
    pieces = gname.split('-')
    if len(pieces) > 2 and pieces[0] == 'tw' and pieces[1] == 'cooking':
        pieces = pieces[2:]
    if len(pieces) > 1 and (pieces[0] == 'train' or pieces[0] == 'valid' or pieces[0] == 'test'):
        pieces = pieces[1:]
    if len(pieces) >= 2:
        skills = pieces[0]
        gid = pieces[1]
    else:
        skills = ""
        gid = pieces[0]
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
        if len(segments) > 2:   # try chopping off the split- identifier (from TW 1.3 cooking.py generator script)
            if segments[0] == "train" or segments[0] == "valid" or segments[0] == "test":
                split = segments[0]+"-"
                segments = segments[1:]
            else:
                print(f"WARNING: unexpected game name: {game_name}")
                return game_id
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
