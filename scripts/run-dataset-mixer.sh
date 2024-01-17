INPUT_FILE=/afs/cern.ch/work/d/dapullia/public/dune/cnn_approach/lists/es-direction-cut.txt
INPUT_LABEL=/afs/cern.ch/work/d/dapullia/public/dune/cnn_approach/lists/es-direction-lab.txt
OUTPUT_FOLDER=/eos/user/d/dapullia/cnn_approach/emaprod/es-direction-cut/
REMOVE_LABELS="99"
SHUFFLE=1
BALANCE=0
# move to the folder, run and come back to scripts
cd ../python/
python dataset-mixer.py --input_data $INPUT_FILE --input_label $INPUT_LABEL --output_folder $OUTPUT_FOLDER --remove_labels $REMOVE_LABELS --shuffle $SHUFFLE --balance $BALANCE
cd ../scripts

