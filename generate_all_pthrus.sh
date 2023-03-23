ulimit -Sn unlimited

# use TWDATA_DIR env variable, assign default value if not set
: "${TWDATA_DIR:=/ssd2tb/twdata}"
export TWDATA_DIR
#mkdir -p ${TWDATA_DIR}
mkdir -p ${TWDATA_DIR}/combined/

#python generate_playthroughs.py  gata_1 --output-dir ${TWDATA_DIR}/gata/playthru_data/ --do-write --overwrite
#python generate_playthroughs.py  gata_20 --output-dir ${TWDATA_DIR}/gata/playthru_data/ --do-write --overwrite
#python generate_playthroughs.py  gata_100 --output-dir ${TWDATA_DIR}/gata/playthru_data/ --do-write --overwrite
python generate_playthroughs.py  gata_valid --flat-gata --input-dir ${TWDATA_DIR}/gata/games_gata/ --output-dir ${TWDATA_DIR}/gata/playthru_data/ --do-write --overwrite
python generate_playthroughs.py  gata_test --flat-gata --input-dir ${TWDATA_DIR}/gata/games_gata/ --output-dir ${TWDATA_DIR}/gata/playthru_data/ --do-write --overwrite
python generate_playthroughs.py  gata_train --flat-gata --input-dir ${TWDATA_DIR}/gata/games_gata/ --output-dir ${TWDATA_DIR}/gata/playthru_data/ --do-write --overwrite

python generate_playthroughs.py  valid --input-dir ${TWDATA_DIR}/ftwc/games_ftwc/ --output-dir ${TWDATA_DIR}/ftwc/playthru_data/ --do-write --overwrite
python generate_playthroughs.py  test --input-dir ${TWDATA_DIR}/ftwc/games_ftwc/ --output-dir ${TWDATA_DIR}/ftwc/playthru_data/ --do-write --overwrite
python generate_playthroughs.py  train --input-dir ${TWDATA_DIR}/ftwc/games_ftwc/ --output-dir ${TWDATA_DIR}/ftwc/playthru_data/ --do-write --overwrite


cp -a ${TWDATA_DIR}/gata/playthru_data/*.textds ${TWDATA_DIR}/
cp -a ${TWDATA_DIR}/ftwc/playthru_data/*.textds ${TWDATA_DIR}/

#mv ${TWDATA_DIR}/gata_100.textds ${TWDATA_DIR}/gata_train.textds
#mv ${TWDATA_DIR}/gata_train.textds ${TWDATA_DIR}/gata_train.textds

python generate_playthroughs.py none --build-tokenizer --tokenizer-filepath ${TWDATA_DIR}/combined/combined_tokenizer.json

python merge_datasets.py --twdata-dir ${TWDATA_DIR}/ --output-dir ${TWDATA_DIR}/combined/

