ulimit -Sn unlimited

# use TWDATA_DIR env variable, assign default value if not set
: "${TWDATA_DIR:=$HOME/work2/twdata}"
export TWDATA_DIR

TEXTDS_DIR=${TWDATA_DIR}/simpler_textds
MERGED_DIR=${TWDATA_DIR}/simpler_combined
PTHRU_SUBDIR=simpler.playthru_data
#mkdir -p ${TWDATA_DIR}
mkdir -p ${MERGED_DIR}/
mkdir -p ${TEXTDS_DIR}/


#python generate_playthroughs.py  gata_1 --internal-names --output-dir ${TWDATA_DIR}/gata/playthru_data/ --do-write --overwrite
#python generate_playthroughs.py  gata_20 --internal-names --output-dir ${TWDATA_DIR}/gata/playthru_data/ --do-write --overwrite
#python generate_playthroughs.py  gata_100 --internal-names --output-dir ${TWDATA_DIR}/gata/playthru_data/ --do-write --overwrite
python generate_playthroughs.py  gata_valid --internal-names --flat-gata --input-dir ${TWDATA_DIR}/gata/games_gata/ --output-dir ${TWDATA_DIR}/gata/${PTHRU_SUBDIR}/ --do-write --overwrite
python generate_playthroughs.py  gata_test --internal-names --flat-gata --input-dir ${TWDATA_DIR}/gata/games_gata/ --output-dir ${TWDATA_DIR}/gata/${PTHRU_SUBDIR}/ --do-write --overwrite
python generate_playthroughs.py  gata_train --internal-names --flat-gata --input-dir ${TWDATA_DIR}/gata/games_gata/ --output-dir ${TWDATA_DIR}/gata/${PTHRU_SUBDIR}/ --do-write --overwrite

python generate_playthroughs.py  valid --internal-names --input-dir ${TWDATA_DIR}/ftwc/games_ftwc/ --output-dir ${TWDATA_DIR}/ftwc/${PTHRU_SUBDIR}/ --do-write --overwrite
python generate_playthroughs.py  test --internal-names --input-dir ${TWDATA_DIR}/ftwc/games_ftwc/ --output-dir ${TWDATA_DIR}/ftwc/${PTHRU_SUBDIR}/ --do-write --overwrite
python generate_playthroughs.py  train --internal-names --input-dir ${TWDATA_DIR}/ftwc/games_ftwc/ --output-dir ${TWDATA_DIR}/ftwc/${PTHRU_SUBDIR}/ --do-write --overwrite

cp -a ${TWDATA_DIR}/gata/${PTHRU_SUBDIR}/*.textds ${TEXTDS_DIR}/
cp -a ${TWDATA_DIR}/ftwc/${PTHRU_SUBDIR}/*.textds ${TEXTDS_DIR}/

python train_tokenizer.py --pthru-dirname ${PTHRU_SUBDIR} --tokenizer-filepath ${MERGED_DIR}/combined_tokenizer.json

python merge_datasets.py --input-dir ${TEXTDS_DIR}/ --output-dir ${MERGED_DIR}/

