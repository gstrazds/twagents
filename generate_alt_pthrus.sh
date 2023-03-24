ulimit -Sn unlimited

# use TWDATA_DIR env variable, assign default value if not set
: "${TWDATA_DIR:=/ssd2tb/twdata}"
export TWDATA_DIR
#mkdir -p ${TWDATA_DIR}
mkdir -p ${TWDATA_DIR}/alt.combined/
mkdir -p ${TWDATA_DIR}/alt/

python generate_playthroughs.py  gata_valid --internal-names --flat-gata --input-dir ${TWDATA_DIR}/gata/games_gata/ --output-dir ${TWDATA_DIR}/gata/alt.playthru_data/ --do-write --overwrite
python generate_playthroughs.py  gata_test --internal-names --flat-gata --input-dir ${TWDATA_DIR}/gata/games_gata/ --output-dir ${TWDATA_DIR}/gata/alt.playthru_data/ --do-write --overwrite
python generate_playthroughs.py  gata_train --internal-names --flat-gata --input-dir ${TWDATA_DIR}/gata/games_gata/ --output-dir ${TWDATA_DIR}/gata/alt.playthru_data/ --do-write --overwrite

python generate_playthroughs.py  valid --internal-names --input-dir ${TWDATA_DIR}/ftwc/games_ftwc/ --output-dir ${TWDATA_DIR}/ftwc/alt.playthru_data/ --do-write --overwrite
python generate_playthroughs.py  test --internal-names --input-dir ${TWDATA_DIR}/ftwc/games_ftwc/ --output-dir ${TWDATA_DIR}/ftwc/alt.playthru_data/ --do-write --overwrite
python generate_playthroughs.py  train --internal-names --input-dir ${TWDATA_DIR}/ftwc/games_ftwc/ --output-dir ${TWDATA_DIR}/ftwc/alt.playthru_data/ --do-write --overwrite


cp -a ${TWDATA_DIR}/gata/alt.playthru_data/*.textds ${TWDATA_DIR}/alt/
cp -a ${TWDATA_DIR}/ftwc/alt.playthru_data/*.textds ${TWDATA_DIR}/alt/

#mv ${TWDATA_DIR}/gata_100.textds ${TWDATA_DIR}/gata_train.textds
#mv ${TWDATA_DIR}/gata_train.textds ${TWDATA_DIR}/gata_train.textds

python generate_playthroughs.py none --flat-gata --build-alt-tokenizer --tokenizer-filepath ${TWDATA_DIR}/alt.combined/combined_tokenizer.json

python merge_datasets.py --twdata-dir ${TWDATA_DIR}/alt --output-dir ${TWDATA_DIR}/alt.combined/

