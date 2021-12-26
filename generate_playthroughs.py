import os
import os.path
import json
from pathlib import Path
from twutils.playthroughs import generate_playthru, export_playthru, get_games_dir, GamesIndex, playthrough_id
from twutils.playthroughs import _list_game_files

def generate_and_export_pthru(gamename, gamedir, outdir,
                              ptdir=None,   # directory for PT.json files (read if not do_generate, else write)
                              randseed=None, goal_type=None,  # will use default values
                              skip_existing=True,
                              do_generate=True,
                              do_export=True,
                              dry_run=False):   # do everything according to other args, but don't write to disk

    assert do_generate or do_export, f"Please select at least one of do_generate({do_generate}), do_export({do_export})"
    if not ptdir:
        ptdir = outdir
    if gamename.endswith(".z8") or gamename.endswith(".ulx"):
        _gamefile = f"{gamedir}/{gamename}"
        gamename = os.path.splitext(gamename)[0]   # want to use it below without the extension
    else:
        _gamefile = f"{gamedir}/{gamename}.z8"
        if not os.path.exists(_gamefile):
            _gamefile = f"{gamedir}/{gamename}.ulx"

    assert os.path.exists(_gamefile), f"{_gamefile} does not exist [gamedir={gamedir} gamename={gamename}]"

    _pthrufile = f"{outdir}/{gamename}.pthru"
    _ptjson = f"{ptdir}/{gamename}_PT.json"
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

    gamesets = {'extra': None,
                'miniset': 'train',
                'train': 'train', 'valid': 'valid', 'test': 'test',
                'gata_train': 'gata_train', 'gata_valid': 'gata_valid', 'gata_test': 'gata_test',
                }

    def main(args):
        total_files = 0
        games_index = GamesIndex()
        if args.which == 'extra':
            print("Generating playthrough data for games from", args.games_dir)

        if args.export_files:
            print("++ Resaving generated playthough data to files ++")

        if not args.which:
            assert False, "Expected which= one of [extra, train, valid, test, miniset, gata_train, gata_valid, gata_test]"
            exit(1)
        subset = gamesets[args.which]
        if subset:
            if subset.startswith("gata_"):
                basepath = os.getenv('GATA_BASEDIR', '/work2/gstrazds/gata/rl.0.2')
                splitname = subset[len("gata_"):]
            else:
                basepath = os.getenv('TW_GAMES_BASEDIR', '/work2/gstrazds/ftwc/games')
                splitname = subset
            gamespath=get_games_dir(basepath=basepath, splitname=splitname)
            #num_games = rj.scard(rediskey)
            gamenames = games_index.count_and_index_gamefiles(which=subset, dirpath=gamespath)
            if args.which == 'miniset':
                # gamenames = list(gamenames)[0:3]   # just the first 3
                gamenames = ['tw-cooking-recipe3+take3+cut+go6-Z7L8CvEPsO53iKDg']
        else:
            gamespath=args.games_dir
            gamenames = _list_game_files(gamespath)
            #num_games = len(gamenames)
            print("num_games:", len(gamenames), gamenames[0:5])
            if args.start_idx is not None:
                gamenames = gamenames[args.start_idx:args.start_idx+1000]
            #small slice FOR TESTING: gamenames = gamenames[0:5]

        if (args.which).startswith("gata_"):
            destdir = f'{args.output_dir}/{args.which}'
        elif args.which != 'extra':
            destdir = f'{args.output_dir}/{args.which}'
        else:
            assert args.which == 'extra', f"UNEXPECTED: invalid combination of args? {str(args)}"
            destdir = f'{args.output_dir}/{args.which}'

        if not args.do_write:
            dry_run = True
        else:
            dry_run = False
        if destdir and args.do_write:
            Path(destdir).mkdir(parents=True, exist_ok=True)
        for i, gname in enumerate(tqdm(gamenames)):
            if args.export_files:
                playthru = retrieve_playthrough_json(gname, ptdir=destdir, ptid=None)
                if args.do_write:
                    total_files += export_playthru(gname, playthru, destdir=destdir)
            else:
                if args.which == 'extra':
                    print(f"[{i}] {gname}")
                    gname = gname.split('.')[0]   # just in case
                num_steps, playthru = \
                    generate_and_export_pthru(gname,
                                              gamedir=gamespath,
                                              outdir=destdir,
                                              randseed=None, goal_type=None,
                                              skip_existing=True,
                                              do_export=args.do_write,
                                              dry_run=dry_run,
                                        )
                total_files += 1

                print(f"[{i}] PLAYTHROUGH {gname}: steps:{num_steps}")

        if args.export_files:
            print("Total files exported:", total_files)
        else:
            print("Total playthroughs generated:", total_files)


    parser = argparse.ArgumentParser(description="Generate or export playthrough data")
    parser.add_argument("which", choices=('extra', 'train', 'valid', 'test', 'miniset', 'gata_train', 'gata_valid', 'gata_test'))
    parser.add_argument("--games-dir", default="./tw_games/", metavar="PATH",
                                   help="Path to directory of .z8 game files from which to generate playthrough data")
    parser.add_argument("--output-dir", default="./playthrus", metavar="PATH",
                                   help="Path to directory for generated playthrough data")
    parser.add_argument("--start-idx", type=int, default=0, help="offset into the list of games")

    parser.add_argument("--export-files", action='store_true', help="output .pthru files (for use as training/validation/test datasets)")
    parser.add_argument("--do-write", action='store_true')
    args = parser.parse_args()
    main(args)

#NOTE: ulimit -aS unlimited   #by default ulimit -nS  seems to be 1000, which causes an exception