#/bin/bash
module load conda/5.0.1-python3.6
source activate virt_font_recog_conda # to activate my virtual environment

sudo mountimg /data/zenith/user/tmondal/Font_Data/Train_Data_Patch.squashfs /data/zenith/user/tmondal/Font_Data/Train_Data_Patch  # to mount some directory, containing data
sudo mountimg /data/zenith/user/tmondal/Font_Data/Validation_Data_Patch.squashfs /data/zenith/user/tmondal/Font_Data/Validation_Data_Patch # to mount some directory, containing data

python -u /home/tmondal/Python_Projects/Patch_Level/Font_Recognition_Single/train_network_word_level_SingleModel.py --resume "janinah" --model "resnet" --taskname "scanning" --batch_size 500 --number_of_class 3

