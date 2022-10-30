import os
import os.path
import json
from pathlib import Path
from twutils.playthroughs import generate_playthrus, export_playthru, get_games_dir, retrieve_playthrough_json, GamesIndex
from twutils.playthroughs import playthrough_id, normalize_path, make_dsfilepath, _list_game_files
from twutils.twlogic import get_name2idmap
from train_tokenizer import build_tokenizer, save_tokenizer_to_json

def generate_and_export_pthru(gamename, gamedir, outdir,
                              ptdir=None,   # directory for PT.json files (read if not do_generate, else write)
                              gindex : GamesIndex = None,  # overrides gamedir if not None
                              randseed=None, goal_type=None,  # will use default values
                              skip_existing=True,
                              do_generate=True,
                              do_export=True,
                              dry_run=False,  # do everything according to other args, but don't write to disk
                              dataset_name=None,
                              use_internal_names=False):

    assert do_generate or do_export, f"Please select at least one of do_generate({do_generate}), do_export({do_export})"
    if gindex is not None:
        _gamedir = gindex.get_dir_for_game(gamename)
        if _gamedir and gamedir:
            print(f"WARNING: OVERRIDING dir:{gamedir} <- {_gamedir} for {gamename}")
            gamedir = _gamedir
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
        step_array_list, games_list = generate_playthrus([_gamefile], randseed=randseed, use_internal_names=use_internal_names)
        # for step_array in step_array_list:
        step_array = step_array_list[0]
        game = games_list[0]
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
            print(warn_prefix, _pthrufile)
        else:
            warn_prefix = None
        if not warn_prefix or not skip_existing:  # file doesn't exist, or write even if it does
            if use_internal_names:
                if not do_generate:
                    game = None    #TODO: get the game data from somewhere
                assert do_generate, "Reexport with --internal-names is NOT YET supported"
                names2ids = get_name2idmap(game)  # we got the game data from generate_playthrus()
            export_playthru(gamename, step_array, destdir=outdir, dry_run=dry_run, dataset_name=dataset_name, map_names2ids=names2ids)
    return num_steps, step_array


if __name__ == "__main__":
    import argparse
    from tqdm import tqdm

    gamesets = {'extra': None, 'none': None,
                'miniset': 'train',
                'train': 'train', 'valid': 'valid', 'test': 'test',
                'gata_100': 'train_100', 'gata_20': 'train_20', 'gata_1': 'train_1',
                'gata_valid': 'valid', 'gata_test': 'test',
                }

    def main(args):
        total_files = 0
        games_index = None if args.which == 'extra' else GamesIndex()   # not used if target is 'extra'
        if args.which == 'extra':
            print("Generating playthrough data for games from", args.input_dir)

        if args.reexport_pthrus:
            print("++ Resaving generated playthough data to files ++")

        if not args.which:
            assert False, "Expected which= one of [extra, train, valid, test, miniset, gata_100, gata_20, gata_1, gata_valid, gata_test]"
            exit(1)
        subset = gamesets[args.which]   # == None if args.which == 'extra'
        if args.which == 'none':
            print("no playthru processing -- presumably just want to (re)build tokenizer?")
            gamenames = []
        elif args.which == 'extra':
            gamespath=args.input_dir
            gamenames = _list_game_files(gamespath)
            #num_games = len(gamenames)
            print("num_games:", len(gamenames), gamenames[0:5])
            if args.start_idx is not None:
                gamenames = gamenames[args.start_idx:args.start_idx+1000]
            #small slice FOR TESTING: gamenames = gamenames[0:5]
        else:
            difficulty_prefix = ''
            levels = [0]  # FTWC doesn't segregate games by difficulty levels
            if (args.which).startswith("gata_"):
                if not args.reexport_pthrus:
                    difficulty_prefix = 'difficulty_level_{level:d}'
                    levels = list(range(1,11))   # 10 difficulty levels, numbered from 1 to 10
                if not args.input_dir:
                    basepath = os.getenv('GATA_BASEDIR', '/work2/gstrazds/twdata/gata/rl.0.2')
                else:
                    basepath = args.input_dir
                splitname = subset
            else:  #by default we use FTWC games

                if not args.input_dir:
                    # TW_TRAINING_DIR = get_games_dir(basepath=TW_GAMES_BASEDIR,splitname='train')  # '/ssd2tb/ftwc/games/train/'
                    # assert os.path.exists(TW_TRAINING_DIR)
                    # TW_VALIDATION_DIR = get_games_dir(basepath=TW_GAMES_BASEDIR,splitname='valid')  # '/ssd2tb/ftwc/games/valid/'
                    # assert os.path.exists(TW_VALIDATION_DIR)
                    # TW_TEST_DIR = get_games_dir(basepath=TW_GAMES_BASEDIR,splitname='test')  # '/ssd2tb/ftwc/games/test/'
                    # assert os.path.exists(TW_TEST_DIR)
                    basepath = os.getenv('TW_GAMES_BASEDIR', '/work2/gstrazds/twdata/ftwc/games_ftwc')
                else:
                    basepath = args.input_dir
                splitname = subset
            subsetpath = normalize_path(basepath, splitname)
            gamenames = []
            for level in levels:
                if difficulty_prefix:
                    subdir = difficulty_prefix.format(level=level)
                else:
                    subdir = None
                if args.reexport_pthrus:
                    # gamespath = normalize_path(basepath, args.which)
                    gamenames.extend(games_index.count_and_index_pthrus(which=splitname, dirpath=basepath))
                else:
                    gamespath = normalize_path(subsetpath, subdir)
                    gamenames.extend(games_index.count_and_index_gamefiles(which=splitname, dirpath=gamespath))
            if args.which == 'miniset':
                gamenames = list(gamenames)[0:3]   # just the first 3
                gamenames.extend(['tw-cooking-recipe3+take3+cut+go6-Z7L8CvEPsO53iKDg']) #plus one specifically selected

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
            Path(make_dsfilepath(destdir, args.which)).unlink(missing_ok=True)
        if gamenames:
            for i, gname in enumerate(tqdm(gamenames)):
                if args.reexport_pthrus:
                    _output_dir = args.output_dir
                    if not _output_dir:
                        FTWC_PTHRU = os.getenv('FTWC_PTHRU',
                                               '/work2/gstrazds/ftwc/playthru_data')  # '/ssd2tb/ftwc/playthru_data')
                        assert os.path.exists(FTWC_PTHRU)

                        FTWC_TRAIN_PTHRUS = normalize_path(FTWC_PTHRU, 'train')  # '/ssd2tb/ftwc/playthru_data/train/'
                        FTWC_VALID_PTHRUS = normalize_path(FTWC_PTHRU, 'valid')  # '/ssd2tb/ftwc/playthru_data/valid/'
                        FTWC_TEST_PTHRUS = normalize_path(FTWC_PTHRU, 'test')  # '/ssd2tb/ftwc/playthru_data/test/'

                    playthru = retrieve_playthrough_json(gname, ptdir=args.input_dir, gindex=games_index, ptid=None)
                    if args.do_write:
                        if args.internal_names:
                            assert False, "reexport with --internal-names NOT YET SUPPORTED"
                            _game_ = None   # TODO: need to retrieve game data from somewhere (e.g. gamefile)
                            names2ids = get_name2idmap(_game_)
                        total_files += export_playthru(gname, playthru, destdir=destdir, dataset_name=args.which, map_names2ids=names2ids)
                else:
                    if args.which == 'extra':
                        print(f"[{i}] {gname}")
                        gname = gname.split('.')[0]   # just in case
                    num_steps, playthru = generate_and_export_pthru(gname,
                                                  gamedir=gamespath,
                                                  gindex=games_index,
                                                  outdir=destdir,
                                                  randseed=None, goal_type=None,
                                                  skip_existing=(not args.overwrite),
                                                  do_export=args.do_write,
                                                  dry_run=dry_run,
                                                  dataset_name=args.which,
                                                  use_internal_names=args.internal_names,
                                            )
                    total_files += 1

                    print(f"[{i}] PLAYTHROUGH {gname}: steps:{num_steps}")

        if args.reexport_pthrus:
            print("Total files exported:", total_files)
        else:
            print("Total playthroughs generated:", total_files)

        if args.build_tokenizer:
            if args.tokenizer_input_dirs:
                dir_list = args.tokenizer_input_dirs
            else:
                PTHRU_DIR = '/work2/gstrazds/twdata/ftwc/playthru_data/'
                GATA_PTHRU_DIR = '/work2/gstrazds/twdata/gata/playthru_data/'
                glob_list = [
                    PTHRU_DIR + "valid/*.pthru",
                    PTHRU_DIR + "test/*.pthru",
                    PTHRU_DIR + "train/*.pthru",
                    GATA_PTHRU_DIR + "gata_valid/*.pthru",
                    GATA_PTHRU_DIR + "gata_test/*.pthru",
                    GATA_PTHRU_DIR + "gata_100/*.pthru",
                ]

            tokenizer = build_tokenizer(glob_list)
            save_tokenizer_to_json(tokenizer, args.tokenizer_filepath)


    parser = argparse.ArgumentParser(description="Generate or export playthrough data")
    parser.add_argument("which",
                        choices=('extra', 'none', 'train', 'valid', 'test', 'miniset',
                                 'gata_100', 'gata_20', 'gata_1', 'gata_valid', 'gata_test'))
    parser.add_argument("--input-dir", default=None, metavar="PATH",
                        help="(Optional) Path to directory of .z8 games or _PT.json files from which to generate pthru data")
    parser.add_argument("--output-dir", default="./playthrus", metavar="PATH",
                        help="(Optional) Path to directory for generated playthrough data")
    parser.add_argument("--start-idx", type=int, default=None,
                        help="(Optional) Offset into the list of games - range:[START_IDX:START_IDX+1000]")

    parser.add_argument("--overwrite", action='store_true', help="overwrite instead of skip if output file already exists")
    parser.add_argument("--reexport-pthrus", action='store_true', help="rewrite .pthru files from _PT.json")
    parser.add_argument("--do-write", action='store_true',
                        help="If not specified, dry run (without writing anything to disk)")
    parser.add_argument("--build-tokenizer", action='store_true',
                        help="Train a tokenizer from the generated playthrough files")
    parser.add_argument("--tokenizer-input-dirs", type=list, nargs="+",
                        help="One or more paths to playthrough output-dirs")
    parser.add_argument("--tokenizer-filepath", default="tokenizer_new.json",
                        help="File path to use when saving newly trained tokenizer")
    parser.add_argument("--internal-names", action='store_true',
                        help="Use TextWorld internal ids instead of entity and room names")
    args = parser.parse_args()
    main(args)

#NOTE: ulimit -aS unlimited   #by default ulimit -nS  seems to be 1000, which causes an exception