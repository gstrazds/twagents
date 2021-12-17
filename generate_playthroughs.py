import os
import os.path
import json
from twutils.playthroughs import generate_playthru, export_playthru

def generate_and_export_pthru(gamename, gamedir, outdir,
                              ptdir=None,   # directory for PT.json files (read if not do_generate, else write)
                              randseed=DEFAULT_PTHRU_SEED, goal_type=GOAL_MEAL,
                              skip_existing=True,
                              do_generate=True,
                              do_export=True,
                              dry_run=False):   # do everything according to other args, but don't write to disk

    assert do_generate or do_export, f"Please select at least one of do_generate({do_generate}), do_export({do_export})"
    if not ptdir:
        ptdir = outdir
    if gamename.endswith(".z8") or gamefile.endswith(".ulx"):
        _gamefile = f"{gamedir}/{gamename}"
        gamename = os.path.splitext(gamename)[0]   # want to use it below without the extension
    else:
        _gamefile = f"{gamedir}/{gamename}.z8"
        if not os.path.exists(_gamefile):
            _gamefile = f"{gamedir}/{gamename}.ulx"

    assert os.path.exists(_gamefile), f"{_gamefile} does not exist [gamedir={gamedir} gamename={gamename}]"

    _pthrufile = f"{outdir}/{basename}.pthru"
    _ptjson = f"{ptdir}/{basename}_PT.json"
    ptid = playthrough_id(objective_name=goal_type, seed=randseed)  # playtrough ID (which of potentially different) for this gamename

    if do_generate:
        step_array = generate_playthru(_gamefile, randseed=randseed)
        _jsonstr = json.dumps(step_array, indent=2)
        if os.path.exists(_ptjson):
            warn_prefix = "SKIPPING - PT file exists:" if skip_existing else "WARNING - OVERWRITING PT:"
            print(warn_prefix, _ptjson)
        if not dry_run:
            with open(_ptjson, "w") as outfile:
                outfile.write(_jsonstr + '\n')
    else:
        with open(_ptjson, "r") as infile:
            step_array = json.load(infile)
    num_steps = len(step_array) if step_array else 0
    if do_export:
        if os.path.exists(_pthrufile):
            warn_prefix = "SKIPPING - pthru file exists:" if skip_existing else "WARNING - OVERWRITING pthru:"
            print(warn_prefix, _phtrufile)
        else:
            warn_prefix = None
        if not warn_prefix or not skip_existing:  # file doesn't exist, or write even if it does
            export_playthru(gamename, step_array, destdir=outdir, dry_run=dry_run)
    return num_steps, step_array


if __name__ == "__main__":
    import argparse
    from tqdm import tqdm

    gamesets = {'extra': None, 'train': REDIS_FTWC_TRAINING, 'valid': REDIS_FTWC_VALID, 'test': REDIS_FTWC_TEST,
                'miniset': REDIS_FTWC_TRAINING,
                'gata_train': REDIS_GATA_TRAINING, 'gata_valid': REDIS_GATA_VALID, 'gata_test': REDIS_GATA_TEST,
                }

    def main(args):
        total_files = 0
        if args.which == 'extra':
            print("Generating playthrough data for games from", args.extra_games_dir)

        if args.export_files:
            print("++ Also saving generated playthough data to files ++")

        if not args.which:
            assert False, "Expected which= one of [extra, train, valid, test, miniset, gata_train, gata_valid, gata_test]"
            exit(1)
        rediskey = gamesets[args.which]
        if rediskey:
            if (args.which).startswith("gata_"):
                redisbasekey = REDIS_GATA_PLAYTHROUGHS
            else:
                redisbasekey = REDIS_FTWC_PLAYTHROUGHS
            #num_games = rj.scard(rediskey)
            gamenames = rj.smembers(rediskey)
            if args.which == 'miniset':
                # gamenames = list(gamenames)[0:3]   # just the first 3
                gamenames = ['tw-cooking-recipe3+take3+cut+go6-Z7L8CvEPsO53iKDg']
        else:
            gamenames = _list_games(args.extra_games_dir)
            #num_games = len(gamenames)
            print("num_games:", len(gamenames), gamenames[0:5])
            if args.start_idx is not None:
                gamenames = gamenames[args.start_idx:args.start_idx+1000]
            #small slice FOR TESTING: gamenames = gamenames[0:5]
        if args.export_files:
            if (args.which).startswith("gata_"):
                destdir = f'./playthru_data/{args.which}'
            elif args.which != 'extra':
                destdir = f'./playthru_data/{args.which}'
            elif args.which == 'extra':
                destdir = f'./playthru_extra/'
            else:
                assert False, f"UNEXPECTED: invalid combination of args? {str(args)}"

        for i, gname in enumerate(tqdm(gamenames)):
            if args.which == 'extra':
                print(f"[{i}] {gname}")
                num_steps, playthru = \
                    generate_and_export_pthru(gname.split('.')[0], gamedir=args.extra_games_dir,
                                              outdir,
                                              randseed=DEFAULT_PTHRU_SEED, goal_type=GOAL_MEAL, skip_existing=True,
                                              do_export=args.do_write)
            else:
                if not args.export_files:
                    print(f"[{i}] BEGIN PLAYTHROUGH: {gname}")
                    num_steps, redis_ops, _ = save_playthrough_to_redis(gname, redis=rj,
                                                                     redisbasekey=redisbasekey,
                                                                     do_write=args.do_write)
                    print(f"[{i}] PLAYTHROUGH {gname}: steps:{num_steps} to redis: {redis_ops}")
                    total_redis_ops += redis_ops
                else:
                    playthru = retrieve_playthrough_json(gname, redis=redis, redisbasekey=redisbasekey)
                    total_files += export_playthru(gname, playthru, destdir=destdir)

        print("Total files exported:", total_files)
        if rj:
            if args.do_write and not args.export_files:
                rj.save()
            rj.close()

    parser = argparse.ArgumentParser(description="Generate or export playthrough data")
    parser.add_argument("which", choices=('extra', 'train', 'valid', 'test', 'miniset', 'gata_train', 'gata_valid', 'gata_test'))
    parser.add_argument("--games-dir", default="./tw_games/", metavar="PATH",
                                   help="Path to directory of .z8 game files from which to generate playthrough data")
    parser.add_argument("--output-dir", default="./playthrus/", metavar="PATH",
                                   help="Path to directory for generated playthrough data")
    parser.add_argument("--start-idx", type=int, default=0, help="offset into the list of games")

    parser.add_argument("--export_files", action='store_true', help="output .pthru files (for use as training/validation/test datasets)")
    parser.add_argument("--do_write", action='store_true')
    args = parser.parse_args()
    main(args)

