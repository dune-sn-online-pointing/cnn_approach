INPUT_FILE=/afs/cern.ch/work/d/dapullia/public/dune/cnn_approach/lists/maint-vs-all.txt
INPUT_LABEL=/afs/cern.ch/work/d/dapullia/public/dune/cnn_approach/lists/maint-vs-all-lab.txt
OUTPUT_FOLDER=/eos/user/d/dapullia/cnn_approach/emaprod/3-1-maintrack-vs-all/
REMOVE_LABELS="99"
SHUFFLE=1
BALANCE=1
# move to the folder, run and come back to scripts
cd ../python/
python dataset-mixer.py --input_data $INPUT_FILE --input_label $INPUT_LABEL --output_folder $OUTPUT_FOLDER --remove_labels $REMOVE_LABELS --shuffle $SHUFFLE --balance $BALANCE
cd ../scripts

