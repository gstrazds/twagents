cd ~/work/github/0_magd3/CodaLab/z8_maps/
find . -type f -name "*.z8map" -exec grep -i -A6 "cookbook\":" {} + > ../extracted_data/0_cookbooks.txt
cd ~/work/github/0_magd3/CodaLab/extracted_data/
grep -h "desc" 0_cookbooks.txt > 0_recipes.txt
find . -type f -name "initial_state.state" -exec grep -h -i -A 1 objective: {} + > 0_objectives.txt
#find . -type f -name "initial_state.state" -exec grep -i hungry {} + > 0_objectives2.txt
tw-stats ../../train/*.json > 0_stats.txt