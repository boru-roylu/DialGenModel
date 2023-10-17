#! /bin/bash -xe

TASK=$1
MODEL=$2
RAW_DATA_DIR=$3

if [ $# -ne 3 ]
then
    echo "Usage: $0 <task> <model> <raw_data_dir>"
    echo "task = {tlb, dst, sc}."
    echo "model = {t5, longt5}."
    echo "Examples"
    echo "   1. tlb t5."
    echo "   2. dst longt5."
    echo "   3. sc t5."
    exit 1
fi

PROJ_DIR=`realpath .`
RAW_DATA_DIR=`realpath $RAW_DATA_DIR`
#RAW_DATA_DIR=$RAW_DATA_DIR/$TASK/replace_numbers_False/v1.8

DATA_DIR=$PROJ_DIR/data/${MODEL}-${TASK}
DATA_SETTING_PATH=$PROJ_DIR/metadata/data_setting/${MODEL}-${TASK}_data_setting.yaml

echo "        TASK = $TASK"
echo "       MODEL = $MODEL"
echo "RAW_DATA_DIR = $RAW_DATA_DIR"
echo "    DATA_DIR = $DATA_DIR"

mkdir -p $DATA_DIR

cd $DATA_DIR

mkdir -p train dev test
ln -s $DATA_SETTING_PATH ./data_setting.yaml
cd train
ln -s $RAW_DATA_DIR/train.json
cd ../dev
ln -s $RAW_DATA_DIR/dev.json
cd ../test
ln -s $RAW_DATA_DIR/test.json

cd $PROJ_DIR