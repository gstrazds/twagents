ulimit -Sn unlimited

# use TWDATA_DIR env variable, assign default value if not set
: "${TWDATA_DIR:=/ssd2tb/twdata}"
export TWDATA_DIR
mkdir -p ${TWDATA_DIR}

python generate_playthroughs.py  gata_1 --output-dir ${TWDATA_DIR}/gata/playthru_data/ --do-write --overwrite

python generate_playthroughs.py  valid --output-dir ${TWDATA_DIR}/ftwc/playthru_data/ --do-write --overwrite
python generate_playthroughs.py  test --output-dir ${TWDATA_DIR}/ftwc/playthru_data/ --do-write --overwrite
python generate_playthroughs.py  train --output-dir ${TWDATA_DIR}/ftwc/playthru_data/ --do-write --overwrite

python generate_playthroughs.py  gata_valid --output-dir ${TWDATA_DIR}/gata/playthru_data/ --do-write --overwrite
python generate_playthroughs.py  gata_test --output-dir ${TWDATA_DIR}/gata/playthru_data/ --do-write --overwrite
python generate_playthroughs.py  gata_100 --output-dir ${TWDATA_DIR}/gata/playthru_data/ --do-write --overwrite
python generate_playthroughs.py  gata_20 --output-dir ${TWDATA_DIR}/gata/playthru_data/ --do-write --overwrite

cp -a ${TWDATA_DIR}/gata/playthru_data/*.textds ${TWDATA_DIR}/
cp -a ${TWDATA_DIR}/ftwc/playthru_data/*.textds ${TWDATA_DIR}/

mv ${TWDATA_DIR}/gata_100.textds ${TWDATA_DIR}/gata_train.textds

python generate_playthroughs.py none --build-tokenizer --tokenizer-filepath ${TWDATA_DIR}/combined/combined_tokenizer.json

python merge_datasets.py --twdata-dir ${TWDATA_DIR}/ --output-dir ${TWDATA_DIR}/combined/

