from collections import OrderedDict

import torch
import torch.nn as nn

from torchvision import transforms
from word_image_datasets import WordImageDS

from network_model import VariousModels, SingleOutputModel


def main():
    test_imag_paths = "/data/zenith/user/tmondal/Font_Data/Test_Data_Patch/"

    saved_model_path_scan = "/home/tmondal/Python_Projects/Patch_Level/Font_Recognition_Single/checkpoint/" \
                            "resnetscanning/Models_epoch_4.ckpt"
    saved_model_path_emphasis = "/home/tmondal/Python_Projects/Patch_Level/Font_Recognition_Single/checkpoint/" \
                                "resnetfont_emphasis/Models_epoch_5.ckpt"
    saved_model_path_size = "/home/tmondal/Python_Projects/Patch_Level/Font_Recognition_Single/checkpoint/" \
                            "resnetfont_size/Models_epoch_4.ckpt"
    saved_model_path_type = "/home/tmondal/Python_Projects/Patch_Level/Font_Recognition_Single/checkpoint/" \
                            "resnetfont_type/Models_epoch_15.ckpt"
    network_type = "font_emphasis"  # change here
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

    probabilities = []

    n_len_test = len(mn_dataset_loader_test.dataset)
    print('Total number of test images :', n_len_test)

    # The model initialization
    model_name = "resnet"
    num_classes = 4   # change here
    feature_extract = True

    my_model = VariousModels(model_name, num_classes, feature_extract)
    model_ft, input_size = my_model.initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 512)

    dd = .1
    model = SingleOutputModel(model_ft, dd, num_classes)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model.cuda())
        print('model and cuda mixing done')

    model = model.to(device)
    # end of model initialization

    checkpoint = torch.load(saved_model_path_emphasis)  # change here
    model.load_state_dict(checkpoint['state_dict'])

    model.eval()  # Set model to evaluate mode
    get_corrects = 0.0

    for inputs, single_image_label_scans, single_image_label_sizes, single_image_label_types, \
        single_image_label_emphas, single_image_whole_label in mn_dataset_loader_test:

        if network_type == "scanning":
            needed_var = single_image_label_scans
        elif network_type == "font_size":
            needed_var = single_image_label_sizes
        elif network_type == "font_type":
            needed_var = single_image_label_types
        elif network_type == "font_emphasis":
            needed_var = single_image_label_emphas
        elif network_type == "all_labels_together":
            needed_var = single_image_whole_label
        else:
            raise Exception("Sorry, no criterion is matching")

        if is_use_cuda:
            needed_var = needed_var.cuda()
            inputs = inputs.cuda()
            needed_var = needed_var.squeeze()
        else:
            needed_var = needed_var.squeeze()

        outputs = model(inputs)
        get_corrects += torch.sum(torch.max(outputs, 1)[1] == torch.max(needed_var, 1)[1])

        probabilities.extend(outputs)
    variable_acc = get_corrects / n_len_test
    print('Testing Accuracy Font Emphasis : {:4f}'.format(float(variable_acc)))

    probabilitie = []
    probabilities = torch.stack(probabilities)

    # for i in probabilities:
    #     probabilitie.append(i.item())
    # return probabilitie


if __name__ == '__main__':
    main()
