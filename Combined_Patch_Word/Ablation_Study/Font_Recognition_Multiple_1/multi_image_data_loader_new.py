from __future__ import print_function, division
import os
import torch

import glob
import random
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


class WordImageDS(Dataset):
    """
    A customized data loader.
    """

    def __init__(self, images_file_paths_word_level, images_file_paths_patch, num_image_from_class=-1, transform=None):
        """ Intialize the dataset
        """
        # num_image_from_class = -1 means, we should take all the images from all the folders or classes
        # Transforms
        self.to_tensor = transforms.ToTensor()

        self.files_path_patch = images_file_paths_patch
        self.files_path_word_level = images_file_paths_word_level
        self.imageTransformations = transform

        subfolders_first_word_level = [f.path for f in os.scandir(self.files_path_word_level) if f.is_dir()]

        [file_names_word, image_labels_scan_word, image_labels_size_word, image_labels_type_word,
         image_labels_empha_word, keep_only_folder_name] = \
            self.retrieve_word_images_from_folders(subfolders_first_word_level)  # get the word images

        # append all the elements in a single array
        self.keep_all_word_image_paths = []
        self.keep_word_image_path_flag = []
        # self.keep_word_image_transform_type = []  # -1 means no transform

        self.keep_all_word_image_scan_class = []
        self.keep_all_word_image_size_class = []
        self.keep_all_word_image_type_class = []
        self.keep_all_word_image_empha_class = []

        # append all the elements in a single array
        self.keep_all_patch_image_paths = []
        self.keep_patch_image_path_flag = []
        # self.keep_patch_image_transform_type = []  # -1 means no transform

        self.keep_all_patch_image_scan_class = []
        self.keep_all_patch_image_size_class = []
        self.keep_all_patch_image_type_class = []
        self.keep_all_patch_image_empha_class = []

        class_count = 0
        for folder_name_only in list(keep_only_folder_name):

            full_path_folder_patch = images_file_paths_patch + " " + folder_name_only + " " + '/'

            #  get patch images under this folder
            [file_names_patch, image_labels_scan_patch, image_labels_size_patch, image_labels_type_patch,
             image_labels_empha_patch] = self.retrieve_patch_images_from_folders(full_path_folder_patch)

            if (len(file_names_patch) > 0) and (len(file_names_word[class_count]) > 0):

                if len(file_names_word[class_count]) > len(file_names_patch):

                    #  first put the patch images in the array first
                    i_val = 0
                    if num_image_from_class == -1:
                        num_of_patches = len(file_names_patch)
                    else:
                        if len(file_names_patch) > num_image_from_class:
                            num_of_patches = num_image_from_class
                        else:
                            num_of_patches = len(file_names_patch)

                    # for get_names_patch in list(file_names_patch):
                    for xt_patch in range(0, num_of_patches):
                        # self.keep_all_patch_image_paths.append(get_names_patch)
                        self.keep_all_patch_image_paths.append(file_names_patch[i_val])

                        self.keep_patch_image_path_flag.append(0)
                        # self.keep_patch_image_transform_type.append(-1)

                        self.keep_all_patch_image_scan_class.append(image_labels_scan_patch[i_val])
                        self.keep_all_patch_image_size_class.append(image_labels_size_patch[i_val])
                        self.keep_all_patch_image_type_class.append(image_labels_type_patch[i_val])
                        self.keep_all_patch_image_empha_class.append(image_labels_empha_patch[i_val])

                        i_val = i_val + 1

                    # then choose only the more informative word images
                    list_all_nonzeros_stats = []
                    cnt_num = 0
                    for get_names_words in list(file_names_word[class_count]):
                        get_img = cv.imread(get_names_words, 0)
                        get_img = cv.medianBlur(get_img, 5)

                        bin_image = cv.adaptiveThreshold(get_img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                         cv.THRESH_BINARY, 11, 2)
                        # cv.imshow("image", bin_image)
                        # cv.waitKey(0)

                        count_nonzeros = cv.countNonZero(bin_image)
                        list_all_nonzeros_stats.append([count_nonzeros, cnt_num])
                        cnt_num = cnt_num + 1
                    list_all_nonzeros_stats.sort(reverse=True)
                    sort_index = []
                    for xx in list_all_nonzeros_stats:
                        sort_index.append(xx[1])  # getting the second element

                    # get word file names and append them first
                    for xt in range(0, num_of_patches):
                        get_indx = sort_index[xt]
                        get_img_name = file_names_word[class_count][get_indx]

                        self.keep_all_word_image_paths.append(get_img_name)
                        self.keep_word_image_path_flag.append(0)
                        # self.keep_word_image_transform_type.append(-1)

                        self.keep_all_word_image_scan_class.append(image_labels_scan_word[class_count][xt])
                        self.keep_all_word_image_size_class.append(image_labels_size_word[class_count][xt])
                        self.keep_all_word_image_type_class.append(image_labels_type_word[class_count][xt])
                        self.keep_all_word_image_empha_class.append(image_labels_empha_word[class_count][xt])

                else:
                    # get word file names and append them first
                    if num_image_from_class == -1:
                        num_words = len(file_names_word[class_count])
                    else:
                        if len(file_names_word[class_count]) > num_image_from_class:
                            num_words = num_image_from_class
                        else:
                            num_words = len(file_names_word[class_count])

                    # for get_names_words in list(file_names_word[class_count]):
                    i_val = 0
                    for xt_words in range(0, num_words):
                        self.keep_all_word_image_paths.append(file_names_word[class_count][i_val])
                        self.keep_word_image_path_flag.append(0)
                        # sself.keep_word_image_transform_type.append(-1)

                        self.keep_all_word_image_scan_class.append(image_labels_scan_word[class_count][i_val])
                        self.keep_all_word_image_size_class.append(image_labels_size_word[class_count][i_val])
                        self.keep_all_word_image_type_class.append(image_labels_type_word[class_count][i_val])
                        self.keep_all_word_image_empha_class.append(image_labels_empha_word[class_count][i_val])

                        i_val = i_val + 1

                    # then choose only the more informative patch images
                    list_all_nonzeros_stats = []
                    cnt_num = 0
                    for get_names_patch in list(file_names_patch):
                        get_img = cv.imread(get_names_patch, 0)
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

                    # get patch file names and append them first
                    for xt in range(0, num_words):
                        get_indx = sort_index[xt]
                        get_patch_img_name = file_names_patch[get_indx]

                        self.keep_all_patch_image_paths.append(get_patch_img_name)
                        self.keep_patch_image_path_flag.append(0)
                        #  self.keep_patch_image_transform_type.append(-1)

                        self.keep_all_patch_image_scan_class.append(image_labels_scan_patch[xt])
                        self.keep_all_patch_image_size_class.append(image_labels_size_patch[xt])
                        self.keep_all_patch_image_type_class.append(image_labels_type_patch[xt])
                        self.keep_all_patch_image_empha_class.append(image_labels_empha_patch[xt])

            # print('The operation of this class is done', class_count)
            class_count = class_count + 1

        # print('Total number of word files are :', len(self.keep_all_word_image_paths))
        # print('Total number of patch files are :', len(self.keep_all_patch_image_paths))

        for i_tensor in range(len(self.keep_all_word_image_scan_class)):
            scan_class_list = [0] * 3
            scan_class_list[self.keep_all_word_image_scan_class[i_tensor]] = 1

            size_class_list = [0] * 3
            size_class_list[self.keep_all_word_image_size_class[i_tensor]] = 1

            type_class_list = [0] * 6
            type_class_list[self.keep_all_word_image_type_class[i_tensor]] = 1

            emphasis_class_list = [0] * 4
            emphasis_class_list[self.keep_all_word_image_empha_class[i_tensor]] = 1

            self.keep_all_word_image_scan_class[i_tensor] = scan_class_list
            self.keep_all_word_image_size_class[i_tensor] = size_class_list
            self.keep_all_word_image_type_class[i_tensor] = type_class_list
            self.keep_all_word_image_empha_class[i_tensor] = emphasis_class_list

        #  sanity checking
        if len(self.keep_all_patch_image_paths) != len(self.keep_all_word_image_paths):
            raise Exception("Sorry, the length of these two array should be same")

        #  sanity checking

        # for i_ele in range(len(self.keep_all_patch_image_scan_class)):
        #     if self.keep_all_patch_image_scan_class[i_ele] != self.keep_all_word_image_scan_class[i_ele]:
        #         raise Exception("Sorry, the label of these two elements should be same")
        #
        #     if self.keep_all_patch_image_size_class[i_ele] != self.keep_all_word_image_size_class[i_ele]:
        #         raise Exception("Sorry, the label of these two elements should be same")
        #
        #     if self.keep_all_patch_image_type_class[i_ele] != self.keep_all_word_image_type_class[i_ele]:
        #         raise Exception("Sorry, the label of these two elements should be same")
        #
        #     if self.keep_all_patch_image_empha_class[i_ele] != self.keep_all_word_image_empha_class[i_ele]:
        #         raise Exception("Sorry, the label of these two elements should be same")

        self.num_of_files = len(self.keep_all_patch_image_paths)

    # You must override __getitem__ and __len__
    def __getitem__(self, index):
        """ Get a sample from the dataset
        """
        single_word_image_label_scan = self.keep_all_word_image_scan_class[index]  # default values
        single_word_image_label_size = self.keep_all_word_image_size_class[index]
        single_word_image_label_type = self.keep_all_word_image_type_class[index]
        single_word_image_label_empha = self.keep_all_word_image_empha_class[index]

        single_word_img_path = self.keep_all_word_image_paths[index]  # default values
        get_img_word = Image.open(single_word_img_path)  # Open image
        get_img_word = get_img_word.convert('RGB')

        single_patch_img_path = self.keep_all_patch_image_paths[index]  # default values
        get_img_patch = Image.open(single_patch_img_path)  # Open image

        get_img_patch = get_img_patch.convert('RGB')

        if self.imageTransformations is not None:
            get_img_word = self.imageTransformations(get_img_word)
            get_img_patch = self.imageTransformations(get_img_patch)

        list_of_labels = [torch.from_numpy(np.array(single_word_image_label_scan)),
                          torch.from_numpy(np.array(single_word_image_label_size)),
                          torch.from_numpy(np.array(single_word_image_label_type)),
                          torch.from_numpy(np.array(single_word_image_label_empha))]

        return get_img_word, get_img_patch, list_of_labels[0], list_of_labels[1], list_of_labels[2], list_of_labels[3]

    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return self.num_of_files

    def generate_extra_images(self, orig_file_names, orig_image_scan_labels, orig_image_size_labels,
                              orig_image_type_labels, orig_image_empha_labels, num_image_to_create):

        extra_image_orig_path = []
        extra_image_transform = []
        extra_image_scan_class = []
        extra_image_size_class = []
        extra_image_type_class = []
        extra_image_empha_class = []

        for gener_imag in range(num_image_to_create):
            random_index_choose = random.randint(0, len(orig_file_names) - 1)
            img_path = orig_file_names[random_index_choose]  # default values

            generate_random_transform = random.randint(1, 7)

            extra_image_orig_path.append(img_path)
            extra_image_transform.append(generate_random_transform)

            extra_image_scan_class.append(orig_image_scan_labels[random_index_choose])
            extra_image_size_class.append(orig_image_size_labels[random_index_choose])
            extra_image_type_class.append(orig_image_type_labels[random_index_choose])
            extra_image_empha_class.append(orig_image_empha_labels[random_index_choose])

        return extra_image_orig_path, extra_image_transform, extra_image_scan_class, extra_image_size_class, \
               extra_image_type_class, extra_image_empha_class

    #  The following function helps to copy the left over images into 20th folder
    def rename_folders(self, images_file_paths):
        """ Intialize the dataset
        """
        files_path = images_file_paths

        subfoldersClass = [f.path for f in os.scandir(files_path) if f.is_dir()]  # getting the sub folders class

        for dirnamePrinter in list(subfoldersClass):
            if dirnamePrinter.find(" ") >= 0:  # if a space is found
                newfilename = dirnamePrinter.replace(" ", "")  # convert spaces to nothing
                os.rename(dirnamePrinter, newfilename)

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

    def retrieve_patch_images_from_folders(self, full_dir_path):

        temp_get_images_of_class = []
        temp_keep_image_scan_class = []
        temp_keep_image_size_class = []
        temp_keep_image_type_class = []
        temp_keep_image_empha_class = []

        subfolders_second = [f.path for f in os.scandir(full_dir_path) if f.is_dir()]  # getting the subfolders

        for dirname_2 in list(subfolders_second):
            subfolders_third = [f.path for f in os.scandir(dirname_2) if f.is_dir()]  # getting the subfolders

            for dirname_3 in list(subfolders_third):
                # print(dirname_3)

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

        return temp_get_images_of_class, temp_keep_image_scan_class, temp_keep_image_size_class, \
               temp_keep_image_type_class, temp_keep_image_empha_class

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
