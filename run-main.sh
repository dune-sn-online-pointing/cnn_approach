INPUT_FILE=/eos/user/d/dapullia/tp_dataset/tpstream-cb/dataset_img.npy
INPUT_LABEL=/eos/user/d/dapullia/tp_dataset/tpstream-cb/dataset_lab.npy
SAVE_PATH=/eos/user/d/dapullia/cnn_approach/
MODEL_NAME=model
LOAD_MODEL=true


# if load_model is true then add --load_model
if [ "$LOAD_MODEL" = true ] ; then
    LOAD_MODEL_FLAG="--load_model"
else
    LOAD_MODEL_FLAG=""
fi

python cnn2d_classifier.py --input_data $INPUT_FILE --input_label $INPUT_LABEL --save_path $SAVE_PATH --model_name $MODEL_NAME $LOAD_MODEL_FLAG