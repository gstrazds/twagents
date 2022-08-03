ulimit -Sn unlimited

python generate_playthroughs.py  gata_1 --output-dir /ssd2tb/twdata/gata/playthru_data/ --do-write --overwrite

python generate_playthroughs.py  valid --output-dir /ssd2tb/twdata/ftwc/playthru_data/ --do-write --overwrite
python generate_playthroughs.py  test --output-dir /ssd2tb/twdata/ftwc/playthru_data/ --do-write --overwrite
python generate_playthroughs.py  train --output-dir /ssd2tb/twdata/ftwc/playthru_data/ --do-write --overwrite

python generate_playthroughs.py  gata_valid --output-dir /ssd2tb/twdata/gata/playthru_data/ --do-write --overwrite
python generate_playthroughs.py  gata_test --output-dir /ssd2tb/twdata/gata/playthru_data/ --do-write --overwrite
python generate_playthroughs.py  gata_100 --output-dir /ssd2tb/twdata/gata/playthru_data/ --do-write --overwrite
python generate_playthroughs.py  gata_20 --output-dir /ssd2tb/twdata/gata/playthru_data/ --do-write --overwrite

cp -a /ssd2dtb/twdata/gata/playthru_data/*.textds /ssd2tb/twdata/
cp -a /ssd2dtb/twdata/ftwc/playthru_data/*.textds /ssd2tb/twdata/

mv /ssd2tb/twdata/gata_100.textds /ssd2tb/twdata/gata_train.textds

python generate_playthroughs.py none --build-tokenizer --tokenizer-filepath /ssd2tb/twdata/combined/combined_tokenizer.json

python merge_datasets.py --twdata-dir /ssd2tb/twdata/ --output-dir /ssd2tb/twdata/combined/

