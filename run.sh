#!/bin/bash

current_path=$(pwd)

mkdir ./dataset
ln -s $(pwd)/../phoenix2014/phoenix2014-release $(pwd)/dataset/phoenix2014

cd ./utils

python dataset_preprocess.py --output-res 224x224px --process-image --multiprocessing

cd ../

mkdir -p phoenix2014_data/features
cp -r $current_path/dataset/phoenix2014/phoenix-2014-multisigner/annotations/ phoenix2014_data/annotations
ln -s $current_path/dataset/phoenix2014/phoenix-2014-multisigner/features/fullFrame-224x224px/ phoenix2014_data/features/fullFrame-224x224px

cd $current_path

echo "data resized"


