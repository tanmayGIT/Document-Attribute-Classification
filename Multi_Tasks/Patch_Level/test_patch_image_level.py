from collections import OrderedDict
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.optim import lr_scheduler
from trainer import Trainer
from torchvision import transforms
from word_image_datasets_full_image import WordImageDS

from network_model import VariousModels, MultiOutputModel
from logger import Logger
import re
import os
import os.path
from os import path
import shutil


def main():
    test_imag_paths = "/data/zenith/user/tmondal/Font_Data/Test_Data_Patch/"

    saved_model_path = "/home/tmondal/Python_Projects/Font_Recognition/Patch_Level/Font_Recognition_Multiple/checkpoint/resnet_multi_task/Models_epoch_12.ckpt"
    torch.cuda.empty_cache()

    data_transforms = {
        'test': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    is_use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if is_use_cuda else "cpu")

    test_imgs_obj = WordImageDS(test_imag_paths, transform=data_transforms['test'])
    mn_dataset_loader_test = torch.utils.data.DataLoader(dataset=test_imgs_obj, batch_size=500,
                                                               shuffle=True, num_workers=0, drop_last=True)

    n_len_test = len(mn_dataset_loader_test.dataset)
    print('Total number of test images :', n_len_test)

    # The model initialization
    model_name = "resnet_multi_task"
    num_classes = 2
    feature_extract = True

    my_model = VariousModels(model_name, num_classes, feature_extract)
    model_ft, input_size = my_model.initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 512)

    dd = .1
    model = MultiOutputModel(model_ft, dd)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model.cuda())
        print('model and cuda mixing done')

    model = model.to(device)
    # end of model initialization

    checkpoint = torch.load(saved_model_path)  # change here
    model.load_state_dict(checkpoint['state_dict'])

    model.eval()  # Set model to evaluate mode
    scan_corrects = 0.0
    size_corrects = 0.0
    type_corrects = 0.0
    emphas_corrects = 0.0

    keep_all_comp_img_info = []
    for inputs, single_image_label_scans, single_image_label_sizes, single_image_label_types, \
        single_image_label_emphas, single_comp_img_path in mn_dataset_loader_test:

        if is_use_cuda:
            inputs, single_image_label_scans, single_image_label_sizes, single_image_label_types, \
            single_image_label_emphas = inputs.cuda(), single_image_label_scans.cuda(), \
                                        single_image_label_sizes.cuda(), single_image_label_types.cuda(), \
                                        single_image_label_emphas.cuda()

            single_image_label_scans = single_image_label_scans.squeeze()
            single_image_label_sizes = single_image_label_sizes.squeeze()
            single_image_label_types = single_image_label_types.squeeze()
            single_image_label_emphas = single_image_label_emphas.squeeze()
        else:
            single_image_label_scans = single_image_label_scans.squeeze()
            single_image_label_sizes = single_image_label_sizes.squeeze()
            single_image_label_types = single_image_label_types.squeeze()
            single_image_label_emphas = single_image_label_emphas.squeeze()

        outputs = model(inputs)

        scan_corrects += torch.sum(torch.max(outputs[0], 1)[1] == torch.max(single_image_label_scans, 1)[1])
        size_corrects += torch.sum(torch.max(outputs[1], 1)[1] == torch.max(single_image_label_sizes, 1)[1])
        type_corrects += torch.sum(torch.max(outputs[2], 1)[1] == torch.max(single_image_label_types, 1)[1])
        emphas_corrects += torch.sum(torch.max(outputs[3], 1)[1] == torch.max(single_image_label_emphas, 1)[1])

        for i_batch in range(len(single_comp_img_path)):  # run for each element within the batch
            img_nm = single_comp_img_path[i_batch]

            all_class_pred_prob_scan = outputs[0].cpu().detach().numpy()[i_batch]
            pred_class_prob_scan = torch.max(outputs[0], 1)[0].cpu().detach().numpy()[i_batch]
            pred_class_nm_scan = torch.max(outputs[0], 1)[1].cpu().detach().numpy()[i_batch]
            actual_class_nm_scan = torch.max(single_image_label_scans, 1)[1].cpu().detach().numpy()[i_batch]

            all_class_pred_prob_size = outputs[1].cpu().detach().numpy()[i_batch]
            pred_class_prob_size = torch.max(outputs[1], 1)[0].cpu().detach().numpy()[i_batch]
            pred_class_nm_size = torch.max(outputs[1], 1)[1].cpu().detach().numpy()[i_batch]
            actual_class_nm_size = torch.max(single_image_label_sizes, 1)[1].cpu().detach().numpy()[i_batch]

            all_class_pred_prob_type = outputs[2].cpu().detach().numpy()[i_batch]
            pred_class_prob_type = torch.max(outputs[2], 1)[0].cpu().detach().numpy()[i_batch]
            pred_class_nm_type = torch.max(outputs[2], 1)[1].cpu().detach().numpy()[i_batch]
            actual_class_nm_type = torch.max(single_image_label_types, 1)[1].cpu().detach().numpy()[i_batch]

            all_class_pred_prob_emphasis = outputs[3].cpu().detach().numpy()[i_batch]
            pred_class_prob_emphasis = torch.max(outputs[3], 1)[0].cpu().detach().numpy()[i_batch]
            pred_class_nm_emphasis = torch.max(outputs[3], 1)[1].cpu().detach().numpy()[i_batch]
            actual_class_nm_emphasis = torch.max(single_image_label_emphas, 1)[1].cpu().detach().numpy()[i_batch]

            combine_it = [img_nm, [[[all_class_pred_prob_scan], pred_class_prob_scan, pred_class_nm_scan, actual_class_nm_scan],
                          [[all_class_pred_prob_size], pred_class_prob_size, pred_class_nm_size, actual_class_nm_size],
                          [[all_class_pred_prob_type], pred_class_prob_type, pred_class_nm_type, actual_class_nm_type],
                          [[all_class_pred_prob_emphasis],pred_class_prob_emphasis, pred_class_nm_emphasis, actual_class_nm_emphasis]]]

            keep_all_comp_img_info.append(combine_it)

    variable_acc_scan = scan_corrects / n_len_test
    variable_acc_size = size_corrects / n_len_test
    variable_acc_type = type_corrects / n_len_test
    variable_acc_emphas = emphas_corrects / n_len_test

    print('Testing Accuracy Scan component level: {:4f}'.format(float(variable_acc_scan)))
    print('Testing Accuracy Size component level: {:4f}'.format(float(variable_acc_size)))
    print('Testing Accuracy Type component level: {:4f}'.format(float(variable_acc_type)))
    print('Testing Accuracy Emphasis component level: {:4f}'.format(float(variable_acc_emphas)))

    data_grouping = []
    for ii in range(len(keep_all_comp_img_info)):

        comp_img_nm = keep_all_comp_img_info[ii][0]
        get_data_part = keep_all_comp_img_info[ii][1]

        name_with_ext = os.path.basename(comp_img_nm)
        only_file_nm, _ = os.path.splitext(os.path.splitext(name_with_ext)[0])
        # splited_str = re.split('[-,_]', only_file_nm)

        splited_str_imag_name = re.split('[-]', only_file_nm)
        splited_str_last_name = re.split('[_]', splited_str_imag_name[-1])

        get_full_img_name = splited_str_imag_name[0]
        for jj in range(len(splited_str_imag_name)-1):
            if jj > 0:
                get_full_img_name = get_full_img_name + "-" + splited_str_imag_name[jj]
        get_full_img_name = get_full_img_name + "-" + splited_str_last_name[0]
        if ii == 0:
            data_grouping.append([get_full_img_name, [get_data_part]])
        else:
            find_flag = 0
            for tt in range(len(data_grouping)):
                get_saved_img_nm = data_grouping[tt][0]
                if get_saved_img_nm == get_full_img_name:
                    get_saved_data_part = data_grouping[tt][1]
                    get_saved_data_part.append(get_data_part)
                    data_grouping[tt][1] = get_saved_data_part
                    find_flag = 1
                    break
            if find_flag == 0:  # match not found
                data_grouping.append([get_full_img_name, [get_data_part]])

    # Iterate for all the full images now
    scan_correct = 0.0
    size_correct = 0.0
    type_correct = 0.0
    emphasis_corect = 0.0

    check_num_comp = 0
    for tt in range(len(data_grouping)):
        get_full_image_nm = data_grouping[tt][0]
        splited_str_imag_name = re.split('[-]', get_full_image_nm)

        page_number = splited_str_imag_name[1]
        scan_resol = splited_str_imag_name[2]
        font_size = splited_str_imag_name[3]
        font_type = splited_str_imag_name[4]
        font_emphasis = splited_str_imag_name[5]

        if scan_resol == "150":
            scan_class = 0
        elif scan_resol == "300":
            scan_class = 1
        elif scan_resol == "600":
            scan_class = 2
        else:
            raise Exception('scan class is not recognizable')

        if font_size == "08":
            font_size_class = 0
        elif font_size == "10":
            font_size_class = 1
        elif font_size == "12":
            font_size_class = 2
        else:
            raise Exception('font size class is not recognizable')

        if font_type == "Ari":
            font_type_class = 0
        elif font_type == "Cal":
            font_type_class = 1
        elif font_type == "Cou":
            font_type_class = 2
        elif font_type == "Tim":
            font_type_class = 3
        elif font_type == "Tre":
            font_type_class = 4
        elif font_type == "Ver":
            font_type_class = 5
        else:
            raise Exception('font type class is not recognizable')

        if font_emphasis == "B":
            font_emphasis_class = 0
        elif font_emphasis == "I":
            font_emphasis_class = 1
        elif font_emphasis == "N":
            font_emphasis_class = 2
        elif font_emphasis == "X":
            font_emphasis_class = 3
        else:
            raise Exception('font type class is not recognizable')

        get_data_part = data_grouping[tt][1]  # number of components present in this full image

        keep_all_class_all_comp_scan = np.empty((0, 3), float)
        keep_all_class_all_comp_size = np.empty((0, 3), float)
        keep_all_class_all_comp_type = np.empty((0, 6), float)
        keep_all_class_all_comp_emphasis = np.empty((0, 4), float)

        for hh in range(len(get_data_part)):  # run for all the component images
            get_each_comp_result = get_data_part[hh]

            get_scan_data = get_each_comp_result[0]
            pred_all_class_prob_scan = get_scan_data[0]
            pred_class_prob_scan = get_scan_data[1]
            pred_class_nm_scan = get_scan_data[2]
            actual_class_nm_scan = get_scan_data[3]

            # keep_all_class_all_comp_scan_2 = np.append(keep_all_class_all_comp_scan,
            # np.array(pred_all_class_prob_scan), axis=0)

            keep_all_class_all_comp_scan = np.append(keep_all_class_all_comp_scan, pred_all_class_prob_scan, axis=0)

            if actual_class_nm_scan != scan_class:
                raise Exception('scan class of this component is not same as full image')

            get_size_data = get_each_comp_result[1]
            pred_all_class_prob_size = get_size_data[0]
            pred_class_prob_size = get_size_data[1]
            pred_class_nm_size = get_size_data[2]
            actual_class_nm_size = get_size_data[3]
            keep_all_class_all_comp_size = np.append(keep_all_class_all_comp_size, pred_all_class_prob_size,
                                                     axis=0)

            if actual_class_nm_size != font_size_class:
                raise Exception('font size class of this component is not same as full image')

            get_type_data = get_each_comp_result[2]
            pred_all_class_prob_type = get_type_data[0]
            pred_class_prob_type = get_type_data[1]
            pred_class_nm_type = get_type_data[2]
            actual_class_nm_type = get_type_data[3]
            keep_all_class_all_comp_type = np.append(keep_all_class_all_comp_type, pred_all_class_prob_type,
                                                     axis=0)

            if actual_class_nm_type != font_type_class:
                raise Exception('font type class of this component is not same as full image')

            get_emphasis_data = get_each_comp_result[3]
            pred_all_class_prob_emphasis = get_emphasis_data[0]
            pred_class_prob_emphasis = get_emphasis_data[1]
            pred_class_nm_emphasis = get_emphasis_data[2]
            actual_class_nm_emphasis = get_emphasis_data[3]
            keep_all_class_all_comp_emphasis = np.append(keep_all_class_all_comp_emphasis,
                                                         pred_all_class_prob_emphasis, axis=0)

            if actual_class_nm_emphasis != font_emphasis_class:
                raise Exception('font emphasis class of this component is not same as full image')

        keep_all_class_all_comp_scan_mean = np.mean(keep_all_class_all_comp_scan, axis=0)
        page_level_scan_class = keep_all_class_all_comp_scan_mean.argmax()
        if page_level_scan_class == scan_class:
            scan_correct = scan_correct + 1

        keep_all_class_all_comp_size_mean = np.mean(keep_all_class_all_comp_size, axis=0)
        page_level_size_class = keep_all_class_all_comp_size_mean.argmax()
        if page_level_size_class == font_size_class:
            size_correct = size_correct + 1

        keep_all_class_all_comp_type_mean = np.mean(keep_all_class_all_comp_type, axis=0)
        page_level_type_class = keep_all_class_all_comp_type_mean.argmax()
        if page_level_type_class == font_type_class:
            type_correct = type_correct + 1

        keep_all_class_all_comp_emphasis_mean = np.mean(keep_all_class_all_comp_emphasis, axis=0)
        page_level_emphasis_class = keep_all_class_all_comp_emphasis_mean.argmax()
        if page_level_emphasis_class == font_emphasis_class:
            emphasis_corect = emphasis_corect + 1

        check_num_comp = check_num_comp + len(get_data_part)    
    
    total_scan_accuracy = scan_correct / len(data_grouping)
    total_size_accuracy = size_correct / len(data_grouping)
    total_type_accuracy = type_correct / len(data_grouping)
    total_emphasis_accuracy = emphasis_corect / len(data_grouping)

    print('Total numbes of pages are : {:4f}'.format(len(data_grouping)))
    print('Total numbes of components are to check : {:4f}'.format(check_num_comp))

    print('Testing Accuracy Scan page level: {:4f}'.format(float(total_scan_accuracy)))
    print('Testing Accuracy Size page level: {:4f}'.format(float(total_size_accuracy)))
    print('Testing Accuracy Type page level: {:4f}'.format(float(total_type_accuracy)))
    print('Testing Accuracy Emphasis page level: {:4f}'.format(float(total_emphasis_accuracy)))

if __name__ == '__main__':
    main()
