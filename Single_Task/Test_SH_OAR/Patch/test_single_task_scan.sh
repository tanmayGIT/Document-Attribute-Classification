#/bin/bash
module load conda/5.0.1-python3.6
source activate virt_font_recog_conda # to activate my virtual environment

sudo mountimg /data/zenith/user/tmondal/Font_Data/Test_Data_Patch.squashfs /data/zenith/user/tmondal/Font_Data/Test_Data_Patch  # to mount some directory, containing data

python -u /home/tmondal/Python_Projects/Patch_Level/Font_Recognition_Single/test_font_patch_single_scan.py
