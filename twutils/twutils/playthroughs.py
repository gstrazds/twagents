TW_TRAINING_DIR = '/ssd2tb/ftwc/games/train/'
TW_VALIDATION_DIR = '/ssd2tb/ftwc/games/valid/'
TW_TEST_DIR = '/ssd2tb/ftwc/games/test/'

REDIS_FTWCv0 = "ftwc:v0"
REDIS_FTWCv2019 = "ftwc:cog2019"

REDIS_EXTRACTED_DATA = f"{REDIS_FTWCv0}:extracted"
REDIS_FTWCv0_TRAINING_GAMES = f'{REDIS_FTWCv0}:training-games'
REDIS_FTWCv2019_TRAINING_GAMES = f'{REDIS_FTWCv2019}:training-games'

REDIS_FTWC_TRAINING = REDIS_FTWCv2019_TRAINING_GAMES
REDIS_FTWC_VALID = f'{REDIS_FTWCv2019}:valid-games'
REDIS_FTWC_TEST = f'{REDIS_FTWCv2019}:test-games'

REDIS_FTWC_PLAYTHROUGHS = f'{REDIS_FTWCv2019}:playthroughs'

REDIS_FTWC_SKILLS_MAP = f"{REDIS_FTWCv2019}:skills:"  # REDIS_FTWC_SKILLS_MAP+{skillname} is a Redis SET of game names
REDIS_FTWC_NSTEPS_MAP = f"{REDIS_FTWC_PLAYTHROUGHS}:nsteps"  # Redis hash. mapping gn -> # playthrough steps
REDIS_FTWC_NSTEPS_INDEX = f"{REDIS_FTWC_PLAYTHROUGHS}:nsteps:"  # +{steps} is a Redis SET of gns

DEFAULT_PTHRU_SEED = 42

def playthrough_id(objective_name='eatmeal', seed=None):
    if seed is None:
        seed = DEFAULT_PTHRU_SEED
    return f"{objective_name}_{seed}"

