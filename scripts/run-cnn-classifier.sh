INPUT_FILE=/eos/user/d/dapullia/cnn_approach/emaprod/es-direction-cut/dataset_img.npy
INPUT_LABEL=/eos/user/d/dapullia/cnn_approach/emaprod/es-direction-cut/dataset_label.npy
OUTPUT_FOLDER=/eos/user/d/dapullia/cnn_approach/emaprod/es-direction-cut/
MODEL_NAME=model
LOAD_MODEL=false
HYPEROPT=true
HP_MAX_EVALS=30
# Function to print help message
print_help() {
    echo "*****************************************************************************"
    echo "Usage: run-classifier.sh -i <input_file> -o <output_folder> [-h]"
    echo "  -i  input file"
    echo "  -o  output folder"
    echo "  -h  print this help message"
    echo "*****************************************************************************"
    exit 0
}

export PYTHONPATH=$PYTHONPATH:/afs/cern.ch/work/d/dapullia/public/dune/cnn_approach/external-libs/lib/python3.9/site-packages


# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -i|--input_file)
            INPUT_FILE="$2"
            shift 2
            ;;
        -o|--OUTPUT_FOLDER)
            OUTPUT_FOLDER="$2"
            shift 2
            ;;
        -h|--help)
            print_help
            ;;
        *)
            shift
            ;;
    esac
done

# stop execution if fundamental variables are not set
if [ -z "$INPUT_FILE" ] || [ -z "$OUTPUT_FOLDER" ]
then
    echo "Usage: ./run-cnn.sh -i <input_file> -o <output_folder> [-h]"
    exit 0
fi

# if load_model is true then add --load_model
if [ "$LOAD_MODEL" = true ] ; then
    LOAD_MODEL_FLAG="--load_model"
else
    LOAD_MODEL_FLAG=""
fi

# if hyperopt is true then add --hyperopt
if [ "$HYPEROPT" = true ] ; then
    HYPEROPT_FLAG="--hyperopt"
else
    HYPEROPT_FLAG=""
fi

# move to the folder, run and come back to scripts
cd ../python/cnn2d_classifier/
python cnn2d_classifier.py --input_data $INPUT_FILE --input_label $INPUT_LABEL --output_folder $OUTPUT_FOLDER --model_name $MODEL_NAME $LOAD_MODEL_FLAG $HYPEROPT_FLAG --hp_max_evals $HP_MAX_EVALS
cd ../scripts
