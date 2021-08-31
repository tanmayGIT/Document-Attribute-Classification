from collections import OrderedDict

import os

import glob
import re
import os.path as osp
from PIL import Image
import numpy as np

import torch
from torchvision import transforms
from word_image_datasets import WordImageDS
from word_image_dataset_small import WordImageDS as wordSmallDataset


def load_state_dict(model_dir, is_multi_gpu):
    state_dict = torch.load(model_dir, map_location=lambda storage, loc: storage)['state_dict']
    if is_multi_gpu:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        return new_state_dict
    else:
        return state_dict


#  The following function helps to copy the images into 20 folders by equally dividing them

def subdivide_into_folders(images_file_paths):
    """ Intialize the dataset
    """
    files_path = images_file_paths

    subfoldersClass = [f.path for f in os.scandir(files_path) if f.is_dir()]  # getting the subfolders class

    for dirnamePrinter in list(subfoldersClass):
        subfoldersPrinter = [f.path for f in os.scandir(dirnamePrinter) if f.is_dir()]  # getting the subfolders

        for dirnameScanner in list(subfoldersPrinter):
            subfoldersScanner = [f.path for f in os.scandir(dirnameScanner) if f.is_dir()]  # getting the subfolders

            for dirname_all_scan in list(subfoldersScanner):
                print(dirname_all_scan)

                comp_imgs_file_names = glob.glob(osp.join(dirname_all_scan, '*.jpg'))  # getting all files inside
                num_of_files = len(comp_imgs_file_names)

                if num_of_files > 20:

                    # create 20 folders
                    for xfolders in range(1, 21):  # iterate for loop from 1 to 20
                        folder_path = dirname_all_scan + "/" + "Folder" + str(xfolders) + "/"

                        if not os.path.exists(folder_path):
                            os.makedirs(folder_path)

                    images_in_each_folder = num_of_files // 19

                    iiStart = 0
                    folder_cnt = 1
                    while folder_cnt < 20:
                        dest_path = dirname_all_scan + "/" + "Folder" + str(folder_cnt) + "/"

                        for i_image in range(iiStart, iiStart + images_in_each_folder):
                            get_image_orig_path = comp_imgs_file_names[i_image]
                            get_img = Image.open(get_image_orig_path)  # Open image

                            name_with_ext = os.path.basename(get_image_orig_path)
                            dest_img_full_path = dest_path + name_with_ext
                            get_img.save(dest_img_full_path)  # save in the new folder

                            #  for each_img_file_name in list(comp_imgs_file_names):
                            os.remove(get_image_orig_path)  # remove the image

                        folder_cnt = folder_cnt + 1
                        iiStart = iiStart + images_in_each_folder

                    # the remaining files will go into the 20th folder
                    last_dest_path = dirname_all_scan + "/" + "Folder" + str(20) + "/"
                    for i_image in range(iiStart, num_of_files):  # iiStart will hold the last index

                        get_image_orig_path = comp_imgs_file_names[i_image]
                        get_img = Image.open(get_image_orig_path)  # Open image

                        name_with_ext = os.path.basename(get_image_orig_path)
                        dest_img_full_path = last_dest_path + name_with_ext
                        get_img.save(dest_img_full_path)  # save in the new folder

                        #  for each_img_file_name in list(comp_imgs_file_names):
                        os.remove(get_image_orig_path)  # remove the image


#  The following function helps to copy the left over images into 20th folder

def copy_leftover_images_into_folder(images_file_paths):
    """ Intialize the dataset
    """
    files_path = images_file_paths

    subfoldersClass = [f.path for f in os.scandir(files_path) if f.is_dir()]  # getting the subfolders class

    for dirnamePrinter in list(subfoldersClass):
        subfoldersPrinter = [f.path for f in os.scandir(dirnamePrinter) if f.is_dir()]  # getting the subfolders

        for dirnameScanner in list(subfoldersPrinter):
            subfoldersScanner = [f.path for f in os.scandir(dirnameScanner) if f.is_dir()]  # getting the subfolders

            for dirname_all_scan in list(subfoldersScanner):
                print(dirname_all_scan)

                comp_imgs_file_names = glob.glob(osp.join(dirname_all_scan, '*.jpg'))  # getting all files inside
                num_of_files = len(comp_imgs_file_names)

                iiStart = 0
                if num_of_files > 0:
                    # the remaining files will go into the 20th folder
                    last_dest_path = dirname_all_scan + "/" + "Folder" + str(20) + "/"
                    for i_image in range(iiStart, num_of_files):  # iiStart will hold the last index

                        get_image_orig_path = comp_imgs_file_names[i_image]
                        get_img = Image.open(get_image_orig_path)  # Open image

                        name_with_ext = os.path.basename(get_image_orig_path)
                        dest_img_full_path = last_dest_path + name_with_ext
                        get_img.save(dest_img_full_path)  # save in the new folder

                        #  for each_img_file_name in list(comp_imgs_file_names):
                        os.remove(get_image_orig_path)  # remove the image


#  The following function helps to copy the left over images into 20th folder
def rename_folders(images_file_paths):
    """ Intialize the dataset
    """
    files_path = images_file_paths

    subfoldersClass = [f.path for f in os.scandir(files_path) if f.is_dir()]  # getting the sub folders class

    for dirnamePrinter in list(subfoldersClass):
        if dirnamePrinter.find(" ") >= 0:  # if a space is found
           newfilename = dirnamePrinter.replace(" ", "")  # convert spaces to nothing
           os.rename(dirnamePrinter, newfilename)


def create_save_dataset_small():
    # This following part is for generating the data by using the data loader and to save it
    training_imag_paths = "Train_Data_Comp_Small/"
    validation_imag_paths = "Validation_Data_Comp_Small/"

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    train_imgs_obj = wordSmallDataset(training_imag_paths, 200, transform=data_transforms['train'])
    validation_imgs_obj = wordSmallDataset(validation_imag_paths, 100, transform=data_transforms['val'])
    test_imgs_obj = wordSmallDataset(validation_imag_paths, 50, transform=data_transforms['val'])

    mn_dataset_loader_train_scanning = torch.utils.data.DataLoader(dataset=train_imgs_obj,
                                                                   batch_size=20, shuffle=True, num_workers=5)
    mn_dataset_loader_validation_scanning = torch.utils.data.DataLoader(dataset=validation_imgs_obj,
                                                                        batch_size=10, shuffle=True, num_workers=5)
    mn_dataset_loader_testing = torch.utils.data.DataLoader(dataset=test_imgs_obj, batch_size=10, shuffle=True,
                                                            num_workers=5)
    print(train_imgs_obj[0])

    torch.save(mn_dataset_loader_train_scanning, 'mn_dataset_loader_train_scanning_small.pth')
    torch.save(mn_dataset_loader_validation_scanning, 'mn_dataset_loader_validation_scanning_small.pth')
    torch.save(mn_dataset_loader_testing, 'mn_dataset_loader_testings_small.pth')


def create_save_dataset():
    # This following part is for generating the data by using the dataloader and to save it
    training_imag_paths = "/home/mondal/Documents/Dataset/L3i_Text_Copies/Keep_Data/Train_Data_Server/"
    validation_imag_paths = "/home/mondal/Documents/Dataset/L3i_Text_Copies/Keep_Data/Validation_Data_Server/"
    testing_imag_paths = "/home/mondal/Documents/Dataset/L3i_Text_Copies/Keep_Data/Test_Data_Server/"

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    subdivide_into_folders(validation_imag_paths)
    # copy_leftover_images_into_folder(validation_imag_paths)
    # rename_folders(validation_imag_paths)

    train_imgs_obj = WordImageDS(training_imag_paths, transform=data_transforms['train'])
    mn_dataset_loader_train = torch.utils.data.DataLoader(dataset=train_imgs_obj,
                                                          batch_size=200, shuffle=True, num_workers=3)
    torch.save(mn_dataset_loader_train, 'mn_dataset_loader_train.pth')
    del train_imgs_obj

    test_imgs_obj = WordImageDS(testing_imag_paths, transform=data_transforms['test'])
    mn_dataset_loader_testing = torch.utils.data.DataLoader(dataset=test_imgs_obj, batch_size=100, shuffle=True,
                                                            num_workers=3)
    torch.save(mn_dataset_loader_testing, 'mn_dataset_loader_testing.pth')
    del test_imgs_obj

    validation_imgs_obj = WordImageDS(validation_imag_paths, transform=data_transforms['val'])
    mn_dataset_loader_validation = torch.utils.data.DataLoader(dataset=validation_imgs_obj,
                                                               batch_size=100, shuffle=True, num_workers=3)
    torch.save(mn_dataset_loader_validation, 'mn_dataset_loader_validation.pth')
    del validation_imgs_obj


def main():
    create_save_dataset()
    # create_save_dataset_small()


if __name__ == '__main__':
    main()
