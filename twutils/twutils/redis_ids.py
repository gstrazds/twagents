REDIS_FTWCv0 = "ftwc:v0"
REDIS_FTWCv2019 = "ftwc:cog2019"
REDIS_GATA = "gata:v1"

REDIS_EXTRACTED_DATA = f"{REDIS_FTWCv0}:extracted"
REDIS_FTWCv0_TRAINING_GAMES = f'{REDIS_FTWCv0}:training-games'
REDIS_FTWCv2019_TRAINING_GAMES = f'{REDIS_FTWCv2019}:training-games'

# Redis SETs of FTWC game names
REDIS_FTWC_TRAINING = REDIS_FTWCv2019_TRAINING_GAMES
REDIS_FTWC_VALID = f'{REDIS_FTWCv2019}:valid-games'
REDIS_FTWC_TEST = f'{REDIS_FTWCv2019}:test-games'

REDIS_FTWC_PLAYTHROUGHS = f'{REDIS_FTWCv2019}:playthroughs'

REDIS_FTWC_SKILLS_MAP = f"{REDIS_FTWCv2019}:skills:"  # REDIS_FTWC_SKILLS_MAP+{skillname} is a Redis SET of game names
REDIS_FTWC_NSTEPS_MAP = f"{REDIS_FTWC_PLAYTHROUGHS}:nsteps"  # Redis hash. mapping gn -> # playthrough steps
REDIS_FTWC_NSTEPS_INDEX = f"{REDIS_FTWC_PLAYTHROUGHS}:nsteps:"  # +{steps} is a Redis SET of gns

# Redis SETs of GATA game names
REDIS_GATA_TRAINING = f'{REDIS_GATA}:training-games'
REDIS_GATA_VALID = f'{REDIS_GATA}:valid-games'
REDIS_GATA_TEST = f'{REDIS_GATA}:test-games'

REDIS_GATA_PLAYTHROUGHS = f'{REDIS_GATA}:playthroughs'


REDIS_GATA_SKILLS_MAP = f"{REDIS_GATA}:skills:"  # REDIS_GATA_SKILLS_MAP+{skillname} is a Redis SET of game names
REDIS_GATA_DIFFICULTY_MAP = f"{REDIS_GATA}:difficulty:"   # REDIS_GATA_DIFFICULTY_MAP+{int} is a Redis SET of game names
REDIS_GATA_NSTEPS_MAP = f"{REDIS_GATA_PLAYTHROUGHS}:nsteps"  # Redis hash. mapping gn -> # playthrough steps
REDIS_GATA_NSTEPS_INDEX = f"{REDIS_GATA_PLAYTHROUGHS}:nsteps:"  # +{steps} is a Redis SET of game names

# Redis SETs of game names selected for training the minGTP model
# (filtered to exclude +drop skills and games with too many playthrough steps)
REDIS_MINGPT_ALL = f"{REDIS_FTWCv2019}:mingpt-all"
REDIS_MINGPT_TRAINING = f"{REDIS_FTWCv2019}:mingpt-training"
REDIS_MINGPT_VALID = f"{REDIS_FTWCv2019}:mingpt-valid"
REDIS_MINGPT_TEST = f"{REDIS_FTWCv2019}:mingpt-test"

REDIS_GATAGPT_ALL = f"{REDIS_GATA}:mingpt-all"
REDIS_GATAGPT_TRAINING = f"{REDIS_GATA}:mingpt-training"
REDIS_GATAGPT_VALID = f"{REDIS_GATA}:mingpt-valid"
REDIS_GATAGPT_TEST = f"{REDIS_GATA}:mingpt-test"

REDIS_DIR_MAP = 'REDIS_DIR_MAP'
# redserv.hset(REDIS_DIR_MAP, 'TW_TRAINING_DIR', TW_TRAINING_DIR)
# redserv.hset(REDIS_DIR_MAP, 'TW_VALIDATION_DIR', TW_VALIDATION_DIR)
# redserv.hset(REDIS_DIR_MAP, 'TW_TEST_DIR', TW_TEST_DIR)


