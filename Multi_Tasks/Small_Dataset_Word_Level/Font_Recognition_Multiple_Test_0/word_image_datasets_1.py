from __future__ import print_function, division
import os
import torch

import glob
import re
import os.path as osp
from PIL import Image
import numpy as np

import cv2 as cv

import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms

# Ignore warnings
import warnings

warnings.filterwarnings("ignore")

plt.ion()  # interactive mode

# In this code, we are trying to write another way for data loading (word images). Because the previous data loading is not working properly (may be) and here I thought 
# that if I can write the data loader differently (like the one was done for simple multi task learning for word images), it may work better 

class WordImageDS(Dataset):
    """
    A customized data loader.
    """

    def __init__(self, images_file_paths, num_image_from_class=-1, transform=None):
        """ Intialize the dataset
        """
        # Transforms
        self.to_tensor = transforms.ToTensor()

        self.files_path = images_file_paths
        self.file_names = []
        # (3) Or you can still compose them like
        self.imageTransformations = transform

        self.image_labels_scan = []
        self.image_labels_size = []
        self.image_labels_type = []
        self.image_labels_empha = []
        # self.keep_whole_label = []

        print("Im am inside WordImageDS")
        print("Image file path is :", images_file_paths)
        # print("Image number of images to take from each class are :", num_image_from_class)

        subfolders_first = [f.path for f in os.scandir(self.files_path) if f.is_dir()]  # getting the sub folders class
        file_cnt = 0
        # print("Total number of classes are :", len(subfolders_first))
        for dirname_1 in list(subfolders_first):
            subfolders_second = [f.path for f in os.scandir(dirname_1) if f.is_dir()]  # getting the sub folders

            # getting the images from this folder or class only
            temp_file_names = []
            temp_labels_scan = []
            temp_labels_size = []
            temp_labels_type = []
            temp_labels_empha = []

            for dirname_2 in list(subfolders_second):
                subfolders_third = [f.path for f in os.scandir(dirname_2) if f.is_dir()]  # getting the sub folders

                for dirname_3 in list(subfolders_third):
                    subfolders_fourth = [f.path for f in os.scandir(dirname_3) if f.is_dir()]

                    for dirname_4 in list(subfolders_fourth):
                        comp_imgs_file_names = glob.glob(osp.join(dirname_4, '*.jpg'))  # getting all files inside

                        for each_img_file_name in list(comp_imgs_file_names):
                            name_with_ext = os.path.basename(each_img_file_name)
                            only_file_nm, _ = os.path.splitext(os.path.splitext(name_with_ext)[0])

                            splited_str = re.split('[-,_]', only_file_nm)

                            temp_file_names.append(each_img_file_name)
                            splited_str.reverse()
                            get_class = int(splited_str[0])

                            scan_class, size_class, type_class, emphas_class = self.decide_the_different_class(
                                get_class)
                            scan_class_list = [0] * 3
                            scan_class_list[scan_class] = 1

                            size_class_list = [0] * 3
                            size_class_list[size_class] = 1

                            type_class_list = [0] * 6
                            type_class_list[type_class] = 1

                            emphasis_class_list = [0] * 4
                            emphasis_class_list[emphas_class] = 1

                            temp_labels_scan.append(scan_class_list)
                            temp_labels_size.append(size_class_list)
                            temp_labels_type.append(type_class_list)
                            temp_labels_empha.append(emphasis_class_list)

            # when you got all the images of this class then choose either few only from them or take all the
            # images
            # get word file names and append them first
            if num_image_from_class == -1:
                num_words = len(temp_file_names)
            else:
                if len(temp_file_names) > num_image_from_class:
                    num_words = num_image_from_class
                else:
                    num_words = len(temp_file_names)

            # print("The number of images exists in this class are :", len(temp_file_names))
            # print("The number of images to be considered from this class  :", num_words)

            # then choose only the more informative word images
            list_all_nonzeros_stats = []
            cnt_num = 0
            for get_names_words in list(temp_file_names):
                get_img = cv.imread(get_names_words, 0)
                get_img = cv.medianBlur(get_img, 5)

                bin_image = cv.adaptiveThreshold(get_img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                    cv.THRESH_BINARY, 11, 2)

                count_nonzeros = cv.countNonZero(bin_image)
                list_all_nonzeros_stats.append([count_nonzeros, cnt_num])
                cnt_num = cnt_num + 1
            list_all_nonzeros_stats.sort(reverse=True)
            sort_index = []
            for xx in list_all_nonzeros_stats:
                sort_index.append(xx[1])  # getting the second element

            # print("Before adding this class, the image count was :", file_cnt)

            # get word file names and append them first
            for xt in range(0, num_words):
                get_indx = sort_index[xt]
                get_img_name = temp_file_names[get_indx]

                self.file_names.append(get_img_name)

                self.image_labels_scan.append(temp_labels_scan[xt])
                self.image_labels_size.append(temp_labels_size[xt])
                self.image_labels_type.append(temp_labels_type[xt])
                self.image_labels_empha.append(temp_labels_empha[xt])

                file_cnt = file_cnt +1
            # print("After adding this class, the image count is now :", file_cnt)

            temp_labels_scan.clear()
            temp_labels_size.clear()
            temp_labels_type.clear()
            temp_labels_empha.clear()

        self.num_of_files = len(self.image_labels_scan)
        print("Total number of image files are :", self.num_of_files)
        print("Total number of image files are counted differently :", file_cnt)
    

    # You must override __getitem__ and __len__
    def __getitem__(self, index):
        """ Get a sample from the dataset
        """
        single_image_label_scan1 = self.image_labels_scan[index]  # default values
        single_image_label_size1 = self.image_labels_size[index]
        single_image_label_type1 = self.image_labels_type[index]
        single_image_label_empha1 = self.image_labels_empha[index]

        single_comp_img_path = self.file_names[index]  # default values
        get_img = Image.open(single_comp_img_path)  # Open image
        get_img = get_img.convert('RGB')

        if self.imageTransformations is not None:
            get_img = self.imageTransformations(get_img)

        list_of_labels = [torch.from_numpy(np.array(single_image_label_scan1)),
                          torch.from_numpy(np.array(single_image_label_size1)),
                          torch.from_numpy(np.array(single_image_label_type1)),
                          torch.from_numpy(np.array(single_image_label_empha1))]

        return get_img, list_of_labels[0], list_of_labels[1], list_of_labels[2], list_of_labels[3]

    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return self.num_of_files

    def calculate_mean_std_images(self, mn_dataset_loader):
        mean = 0.0
        for images, _ in mn_dataset_loader:
            batch_samples = images.size(0)
            images = images.view(batch_samples, images.size(1), -1)
            mean += images.mean(2).sum(0)
        mean = mean / len(mn_dataset_loader.dataset)

        var = 0.0
        for images, _ in mn_dataset_loader:
            batch_samples = images.size(0)
            images = images.view(batch_samples, images.size(1), -1)
            var += ((images - mean.unsqueeze(1)) ** 2).sum([0, 2])
        std = torch.sqrt(var / (len(mn_dataset_loader.dataset) * 224 * 224))

        return mean, std

    def decide_the_different_class(self, get_class):
        get_class = int(get_class)
        arr_bold = np.arange(0, 212 + 4, 4)  # to also include the end point
        arr_italic = np.arange(1, 213 + 4, 4)
        arr_none = np.arange(2, 214 + 4, 4)
        arr_bold_italic = np.arange(3, 215 + 4, 4)

        if 0 <= get_class <= 71:
            scan_class = 0  # it defines scanning class of 150
        elif 72 <= get_class <= 143:
            scan_class = 1  # it defines scanning class of 300
        elif 144 <= get_class <= 215:
            scan_class = 2  # it defines scanning class of 600
        else:
            raise Exception('we should have found at least some class')

        if 0 <= get_class <= 23 or 72 <= get_class <= 95 or 144 <= get_class <= 167:
            size_class = 0  # it defines font size class of having the size of 08
        elif 24 <= get_class <= 47 or 96 <= get_class <= 119 or 168 <= get_class <= 191:
            size_class = 1  # it defines font size class of having the size of 10
        elif 48 <= get_class <= 71 or 120 <= get_class <= 143 or 192 <= get_class <= 215:
            size_class = 2  # it defines font size class of having the size of 12
        else:
            raise Exception('we should have found at least some class')

        if 0 <= get_class <= 3 or 24 <= get_class <= 27 or 48 <= get_class <= 51 or 72 <= get_class <= 75 or \
                96 <= get_class <= 99 or 120 <= get_class <= 123 or 144 <= get_class <= 147 or \
                168 <= get_class <= 171 or 192 <= get_class <= 195:
            type_class = 0  # it defines font type class of having the type Arial

        elif 4 <= get_class <= 7 or 28 <= get_class <= 31 or 52 <= get_class <= 55 or 76 <= get_class <= 79 or \
                100 <= get_class <= 103 or 124 <= get_class <= 127 or 148 <= get_class <= 151 or \
                172 <= get_class <= 175 or 196 <= get_class <= 199:
            type_class = 1  # it defines font type class of having the type Calibri

        elif 8 <= get_class <= 11 or 32 <= get_class <= 35 or 56 <= get_class <= 59 or 80 <= get_class <= 83 or \
                104 <= get_class <= 107 or 128 <= get_class <= 131 or 152 <= get_class <= 155 or \
                176 <= get_class <= 179 or 200 <= get_class <= 203:
            type_class = 2  # it defines font type class of having the type Courier

        elif 12 <= get_class <= 15 or 36 <= get_class <= 39 or 60 <= get_class <= 63 or 84 <= get_class <= 87 or \
                108 <= get_class <= 111 or 132 <= get_class <= 135 or 156 <= get_class <= 159 or \
                180 <= get_class <= 183 or 204 <= get_class <= 207:
            type_class = 3  # it defines font type class of having the type Times new roman

        elif 16 <= get_class <= 19 or 40 <= get_class <= 43 or 64 <= get_class <= 67 or 88 <= get_class <= 91 or \
                112 <= get_class <= 115 or 136 <= get_class <= 139 or 160 <= get_class <= 163 or \
                184 <= get_class <= 187 or 208 <= get_class <= 211:
            type_class = 4  # it defines font type class of having the type Trebuchet

        elif 20 <= get_class <= 23 or 44 <= get_class <= 47 or 68 <= get_class <= 71 or 92 <= get_class <= 95 or \
                116 <= get_class <= 119 or 140 <= get_class <= 143 or 164 <= get_class <= 167 or \
                188 <= get_class <= 191 or 212 <= get_class <= 215:
            type_class = 5  # it defines font type class of having the type Verdana

        else:
            raise Exception('we should have found at least some class')

        if get_class in arr_bold:
            emphas_class = 0  # it defines font type class of having the type bold
        elif get_class in arr_italic:
            emphas_class = 1  # it defines font type class of having the type italic
        elif get_class in arr_none:
            emphas_class = 2  # it defines font type class of having the type none
        elif get_class in arr_bold_italic:
            emphas_class = 3  # it defines font type class of having the type bold italic
        else:
            raise Exception('we should have found at least some class')

        return scan_class, size_class, type_class, emphas_class

    def retrieve_word_images_from_folders(self, sub_folder_dir):

        keep_all_image_names_each_class = []
        keep_only_folder_name = []
        keep_image_scan_class = []
        keep_image_size_class = []
        keep_image_type_class = []
        keep_image_empha_class = []

        for dirname_1 in list(sub_folder_dir):

            dir_divide = os.path.basename(os.path.normpath(dirname_1))

            temp_get_images_of_class = []
            temp_keep_image_scan_class = []
            temp_keep_image_size_class = []
            temp_keep_image_type_class = []
            temp_keep_image_empha_class = []

            subfolders_second = [f.path for f in os.scandir(dirname_1) if f.is_dir()]  # getting the subfolders

            for dirname_2 in list(subfolders_second):
                subfolders_third = [f.path for f in os.scandir(dirname_2) if f.is_dir()]  # getting the subfolders

                for dirname_3 in list(subfolders_third):
                    subfolders_fourth = [f.path for f in os.scandir(dirname_3) if f.is_dir()]

                    if not subfolders_fourth:  # check if it is a empty list
                        comp_imgs_file_names = glob.glob(osp.join(dirname_3, '*.jpg'))  # getting all files inside

                        for each_img_file_name in list(comp_imgs_file_names):
                            name_with_ext = os.path.basename(each_img_file_name)
                            only_file_nm, _ = os.path.splitext(os.path.splitext(name_with_ext)[0])
                            splited_str = re.split('[-,_]', only_file_nm)

                            temp_get_images_of_class.append(each_img_file_name)
                            splited_str.reverse()
                            get_class = int(splited_str[0])

                            scan_class, size_class, type_class, emphas_class = self.decide_the_different_class(
                                get_class)

                            temp_keep_image_scan_class.append(scan_class)
                            temp_keep_image_size_class.append(size_class)
                            temp_keep_image_type_class.append(type_class)
                            temp_keep_image_empha_class.append(emphas_class)
                    else:
                        for dirname_4 in list(subfolders_fourth):
                            comp_imgs_file_names = glob.glob(osp.join(dirname_4, '*.jpg'))  # getting all files inside

                            for each_img_file_name in list(comp_imgs_file_names):
                                name_with_ext = os.path.basename(each_img_file_name)
                                only_file_nm, _ = os.path.splitext(os.path.splitext(name_with_ext)[0])
                                splited_str = re.split('[-,_]', only_file_nm)

                                temp_get_images_of_class.append(each_img_file_name)
                                splited_str.reverse()
                                get_class = int(splited_str[0])

                                scan_class, size_class, type_class, emphas_class = self.decide_the_different_class(
                                    get_class)

                                temp_keep_image_scan_class.append(scan_class)
                                temp_keep_image_size_class.append(size_class)
                                temp_keep_image_type_class.append(type_class)
                                temp_keep_image_empha_class.append(emphas_class)

            keep_all_image_names_each_class.append(temp_get_images_of_class)
            keep_image_scan_class.append(temp_keep_image_scan_class)
            keep_image_size_class.append(temp_keep_image_size_class)
            keep_image_type_class.append(temp_keep_image_type_class)
            keep_image_empha_class.append(temp_keep_image_empha_class)
            keep_only_folder_name.append(dir_divide)

        return keep_all_image_names_each_class, keep_image_scan_class, keep_image_size_class, keep_image_type_class, \
               keep_image_empha_class, keep_only_folder_name
