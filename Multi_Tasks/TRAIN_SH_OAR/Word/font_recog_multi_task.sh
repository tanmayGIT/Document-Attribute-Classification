#/bin/bash
module load conda/5.0.1-python3.6
source activate virt_font_recog_conda # to activate my virtual environment

sudo mountimg /data/zenith/user/tmondal/Font_Data/Train_Data_Server.squashfs /data/zenith/user/tmondal/Font_Data/Train_Data_Server  # to mount some directory, containing data
sudo mountimg /data/zenith/user/tmondal/Font_Data/Validation_Data_Server.squashfs /data/zenith/user/tmondal/Font_Data/Validation_Data_Server # to mount some directory, containing data

python -u /home/tmondal/Python_Projects/Font_Recognition/Word_Level/Font_Recognition_Multiple/train_network.py
