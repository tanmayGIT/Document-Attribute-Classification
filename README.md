# Font Size, Type, Emphasis and Scanning Resolution Classification

## Single Task Learning (STL)
The following code is for single task learning. We have performed the operation by using either word images or patch images. 
<br/>
### **STL Word (on complete dataset)** :
      
* See the file for training : Single_Task/Word_Level/**train_network_word_level_SingleModel.py**

* See the file for network architecture : Single_Task/Word_Level/**network_model_Single.py**

* See the file for actual training using loop and epochs : Single_Task/Word_Level/**SingleNetworkTrainer.py**

⮕ The weights and parameters are saved in : **"/checkpoint"** folder

⮕ After training the network and saving the best model in **"/checkpoint"** folder, you can run the testing code. The testing code is in : 

* Single_Task/Word_Level/**test_word_level_SingleModel.py**

⮕ The results are shown in Table. 1, 1<sup>st</sup> row of the paper. Please see the paper for more details.  

<br/>

### **STL Patch (on complete dataset)** :
    
* See the file for training :  Single_Task/Patch_Level/**train_network.py**

* See the file for network architecture :  Single_Task/Patch_Level/**network_model.py**

* See the file for actual training using loop and epochs :  Single_Task/Patch_Level/**trainer.py**    

⮕ The weights and parameters are saved in : **"/checkpoint"** folder

⮕ After training the network and saving the best model in **"/checkpoint"** folder, you can run the testing code. The testing code is in : 

* Single_Task/Patch_Level/  
    * **test_font_patch_single_empha.py**
    * **test_font_patch_single_scan.py**
    * **test_font_patch_single_size.py**
    * **test_font_patch_single_type.py**

⮕ The results are shown in Table. 1, 2<sup>nd</sup> row of the paper. Please see the paper for more details. 

<br/>

## Multi Task Learning (MTL)
The following code is for multi task learning. We have performed the operation by using either word images or patch images. 

<br/>

###  **MTL Word (on complete dataset)** :

* See the file for training :  Multi_Tasks/Word_Level/**train_network.py**

* See the file for network architecture :  Multi_Tasks/Word_Level/**network_model.py**

* See the file for actual training using loop and epochs :  Multi_Tasks/Word_Level/**trainer.py**

⮕ The weights and parameters are saved in : **"/checkpoint"** folder

⮕ After training the network and saving the best model in **"/checkpoint"** folder, you can run the testing code. The testing code is in : 

* Multi_Tasks/Word_Level/**test_word_multiple_all.py**

⮕ The results are shown in Table. 1, 3<sup>rd</sup> row of the paper. Please see the paper for more details. 

<br/>

<br/>

###  **MTL Word: Word Level with multiple FC layers (on small dataset)** :

* See the file for training :  Multi_Tasks/Small_Dataset_Word_Level/Font_Recognition_Multiple_Test_0/**train_network_1.py**


⮕ After training the network and saving the best model in **"/checkpoint"** folder, you can run the testing code. The testing code is in : 

* Multi_Tasks/Small_Dataset_Word_Level/Font_Recognition_Multiple_Test_0/**test_word_multiple_all.py**

⮕ The results are shown in Table. 2, 1<sup>st</sup> row of the paper. Please see the paper for more details. 

<br/>

###  **MTL Word: Word Level with AlexNet layers (on small dataset)** :

* See the file for training :  Multi_Tasks/Small_Dataset_Word_Level/Font_Recognition_Multiple_Test_0/**train_network_2.py**


⮕ After training the network and saving the best model in **"/checkpoint"** folder, you can run the testing code. The testing code is in : 

* Multi_Tasks/Small_Dataset_Word_Level/Font_Recognition_Multiple_Test_0/**test_word_multiple_all.py**

⮕ The results are shown in Table. 2, 2<sup>nd</sup> row of the paper. Please see the paper for more details.  

<br/>

###  **MTL Word: Word Level with VggNet layers (on small dataset)** :

* See the file for training :  Multi_Tasks/Small_Dataset_Word_Level/Font_Recognition_Multiple_Test_0/**train_network_3.py**


⮕ After training the network and saving the best model in **"/checkpoint"** folder, you can run the testing code. The testing code is in : 

* Multi_Tasks/Small_Dataset_Word_Level/Font_Recognition_Multiple_Test_0/**test_word_multiple_all.py**

⮕ The results are shown in Table. S6, 1<sup>st</sup> row of the paper. Please see the paper for more details.  

<br/>

<br/>


### **MTL Patch (on complete dataset)** :
    
* See the file for training :  Multi_Tasks/Patch_Level/**train_network.py**

* See the file for network architecture :  Multi_Tasks/Patch_Level/**network_model.py**

* See the file for actual training using loop and epochs :  Multi_Tasks/Word_Level/**trainer.py** 

⮕ The weights and parameters are saved in : **"/checkpoint"** folder

⮕ After training the network and saving the best model in **"/checkpoint"** folder, you can run the testing code. The testing code is in : 

* Multi_Tasks/Word_Level/**test_patch_multiple_all.py**

⮕ The results are shown in Table. 1, 4<sup>th</sup> row of the paper. Please see the paper for more details. 
<br/>
<br/>

###  **MTL Patch: Patch Level with multiple FC layers (on small dataset)** :

* See the file for training :  Multi_Tasks/Small_Dataset_Patch_Level/Font_Recognition_Multiple_Test_0/**train_network_1.py**

* See the file for training, the only difference compared to it's counter part i.e. **train_network_1.py** that here we have used **"sampler=torch.utils.data.SubsetRandomSampler"** in dataloader (see line 81-91) to check the results :  Multi_Tasks/Small_Dataset_Patch_Level/Font_Recognition_Multiple_Test_0/**train_network_2.py**


⮕ After training the network and saving the best model in **"/checkpoint"** folder, you can run the testing code. The testing code is in : 

* Multi_Tasks/Small_Dataset_Patch_Level/Font_Recognition_Multiple_Test_0/**test_patch_multiple_all.py**

⮕ The results are shown in Table. 2, 3<sup>rd</sup> row of the paper. Please see the paper for more details. 

<br/>

###  **MTL Patch: Patch Level with AlexNet layers (on small dataset)** :

* See the file for training :  Multi_Tasks/Small_Dataset_Patch_Level/Font_Recognition_Multiple_Test_0/**train_network_patch_3.py**


⮕ After training the network and saving the best model in **"/checkpoint"** folder, you can run the testing code. The testing code is in : 

* Multi_Tasks/Small_Dataset_Patch_Level/Font_Recognition_Multiple_Test_0/**test_patch_multiple_all.py**

⮕ The results are shown in Table. 2, 4<sup>th</sup> row of the paper. Please see the paper for more details.  

<br/>

###  **MTL Patch: Patch Level with VggNet layers (on small dataset)** :

* See the file for training :  Multi_Tasks/Small_Dataset_Patch_Level/Font_Recognition_Multiple_Test_0/**train_network_patch_4.py**


⮕ After training the network and saving the best model in **"/checkpoint"** folder, you can run the testing code. The testing code is in : 

* Multi_Tasks/Small_Dataset_Patch_Level/Font_Recognition_Multiple_Test_0/**test_patch_multiple_all.py**

⮕ The results are shown in Table. S6, 2<sup>nd</sup> row of the paper. Please see the paper for more details.  

<br/>

<br/>



## Multi Task and Multi Instance Learning (performed on smaller dataset)
The following code is for multi task and multi instance learning. We have performed the operation by using the word images and patch images together. From now onwards, all the experiments are performed on the smaller dataset.

<br/>

 ### **MTL Word and Patch Fusioned Together (MTL + MI)**:
  Under this category, we have tried several networks. In the following section, each of these networks are explained: 
  <br/>
  <br/>
  #### **Late Concat Multiple FC Layers**
  
  This network is implemented in the follwing code :
* See the file for training :  Combined_Patch_Word/Font_Recognition_Multiple_4/**train_network_concat_equal_1.py**

* See the file for network architecture "CombineMultiOutputModelConcat" :  Combined_Patch_Word/Font_Recognition_Multiple_4/**network_model.py**

* See the file for actual training using loop and epochs :  Combined_Patch_Word/Font_Recognition_Multiple_4/**trainer.py**

The results can be seen in the Table. 3, 1<sup>st</sup> row of the paper. Please see the paper for more details. 

<br/>

#### **To Overcome Overfitting Problem**
  To overcome the overfitting problem, we have tried several small things as follows :
  
⮕ **Adding more dropout layers and batch normalization layer :**

  This network is implemented in the follwing code :
* See the file for training :  Combined_Patch_Word/Font_Recognition_Multiple_4/**train_network_concat_equal_3.py**

* See the file for network architecture "**CombineMultiOutputModelConcat_DropOut**" :  Combined_Patch_Word/Font_Recognition_Multiple_4/**network_model.py**


⮕ **Adding no dropout layers and batch normalization layer :**

* See the file for network architecture "**CombineMultiOutputModelConcat_NoBatchNorm**" :  Combined_Patch_Word/Font_Recognition_Multiple_4/**network_model.py**

⮕ **Using sampler=torch.utils.data.SubsetRandomSampler in DataLoader :**

* See the file for training :  Combined_Patch_Word/Font_Recognition_Multiple_4/**train_network_concat_equal_3.py**
* See the lines: 97 - 107 in **train_network_concat_equal_3.py**

<br/>

#### **Early Concat Multiple FC Layers**

This network is implemented in the follwing code :
* See the file for training :  Combined_Patch_Word/Font_Recognition_Multiple_4/**train_network_concat_equal_4.py**

* See the file for network architecture "**CombineMultiOutputModelEarlyConcat_1**" :  Combined_Patch_Word/Font_Recognition_Multiple_4/**network_model.py**

* See the file for actual training using loop and epochs :  Combined_Patch_Word/Font_Recognition_Multiple_4/**trainer_2.py**

The results can be seen in the Table. 3, 2<sup>nd</sup> row of the paper. Please see the paper for more details. 


<br/>

#### **Early Concat Less FC Layers\_1**

This network is implemented in the follwing code :
* See the file for training :  Combined_Patch_Word/Font_Recognition_Multiple_4/**train_network_concat_equal_5.py**

* See the file for network architecture "**CombineMultiOutputModelEarlyConcat_2**" :  Combined_Patch_Word/Font_Recognition_Multiple_4/**network_model.py**

* See the file for actual training using loop and epochs :  Combined_Patch_Word/Font_Recognition_Multiple_4/**trainer_3.py**

The results can be seen in the Table. S7, 1<sup>st</sup> row of the paper. Please see the paper for more details. 


<br/>

#### **Early Concat Multiple FC Layers\_2**

This network is implemented in the follwing code :
* See the file for training :  Combined_Patch_Word/Font_Recognition_Multiple_4/**train_network_concat_equal_6.py**

* See the file for network architecture "**CombineMultiOutputModelEarlyConcat_3**" :  Combined_Patch_Word/Font_Recognition_Multiple_4/**network_model.py**

* See the file for actual training using loop and epochs :  Combined_Patch_Word/Font_Recognition_Multiple_4/**trainer_4.py**

The results can be seen in the Table. S7, 2<sup>nd</sup> row of the paper. Please see the paper for more details. 

<br/>

#### **Early concat AlexNet like Conv. layers**

This network is implemented in the follwing code :
* See the file for training :  Combined_Patch_Word/Font_Recognition_Multiple_4/**train_network_concat_equal_7.py**

* See the file for network architecture "**CombineMultiOutputModelConvAlexNet**" :  Combined_Patch_Word/Font_Recognition_Multiple_4/**network_model.py**

* See the file for actual training using loop and epochs :  Combined_Patch_Word/Font_Recognition_Multiple_4/**trainer_5.py**

The results can be seen in the Table.3, 3<sup>rd</sup> row of the paper. Please see the paper for more details. 

<br/>

#### **Early concat VggNet like Conv. layers**

This network is implemented in the follwing code :
* See the file for training :  Combined_Patch_Word/Font_Recognition_Multiple_4/**train_network_concat_equal_8.py**

* See the file for network architecture "**CombineMultiOutputModelConvVggNetSimple**" :  Combined_Patch_Word/Font_Recognition_Multiple_4/**network_model.py**

* See the file for actual training using loop and epochs :  Combined_Patch_Word/Font_Recognition_Multiple_4/**trainer_6.py**

The results can be seen in the Table.S7, 3<sup>rd</sup> row of the paper. Please see the paper for more details. 

<br/>

#### **Early concat AlexNet like Conv. layers (patch and noisy patch)**

This network is implemented in the follwing code :

* See the file for training :  Combined_Patch_Word/Font_Recognition_Multiple_5/**train_network_concat_equal_7.py**

* See the file for network architecture "**CombineMultiOutputModelConvAlexNet**" :  Combined_Patch_Word/Font_Recognition_Multiple_5/**network_model.py**

* See the file for actual training using loop and epochs :  Combined_Patch_Word/Font_Recognition_Multiple_5/**trainer_5.py**

* See the dataset loader file, here we have used: **multi_image_data_loader_new_ver1.py**

The results can be seen in the Table.S8, 1<sup>st</sup> row of the paper. Please see the paper for more details. 

<br/>

#### **Early concat VggNet like Conv. layers (patch and noisy patch)**

This network is implemented in the follwing code :
* See the file for training :  Combined_Patch_Word/Font_Recognition_Multiple_5/**train_network_concat_equal_8.py**

* See the file for network architecture "**CombineMultiOutputModelConvVggNetSimple**" :  Combined_Patch_Word/Font_Recognition_Multiple_5/**network_model.py**

* See the file for actual training using loop and epochs :  Combined_Patch_Word/Font_Recognition_Multiple_5/**trainer_6.py**

* See the dataset loader file, here we have used: **multi_image_data_loader_new_ver1.py**

The results can be seen in the Table.S8, 2<sup>nd</sup> row of the paper. Please see the paper for more details. 
<br/>
<br/>
<br/>
<br/>

## Weighted Multi Task and Multi Instance Learning (performed on smaller dataset)
The following code is for weighted multi task and multi instance learning. We have performed the operation by using the word images and patch images together. 

<br/>

 ### **Late concat multiple FC layers**:

This network is implemented in the follwing code :

* See the file for training :  Combined_Patch_Word/Font_Recognition_Multiple_5/**train_network_weighted_concat_equal.py**

* See the file for network architecture "**CombineMultiOutputModelWeightedConcat1**" :  Combined_Patch_Word/Font_Recognition_Multiple_5/**network_model.py**

* See the file for actual training using loop and epochs :  Combined_Patch_Word/Font_Recognition_Multiple_5/**trainer_weighted_concat.py**


The results can be seen in the Table.4, 1<sup>st</sup> row of the paper. Please see the paper for more details. 

<br/>

 ### **Late concat AlexNet like Conv. layers**:

This network is implemented in the follwing code :

* See the file for training :  Combined_Patch_Word/Font_Recognition_Multiple_5/**train_network_weighted_concat_equal_alexnet.py**

* See the file for network architecture "**CombineMultiOutputModelWeightedConcatAlexNet**" :  Combined_Patch_Word/Font_Recognition_Multiple_5/**network_model.py**

* See the file for actual training using loop and epochs :  Combined_Patch_Word/Font_Recognition_Multiple_5/**trainer_weighted_concat_alexnet.py**


The results can be seen in the Table.4, 2<sup>nd</sup> row of the paper. Please see the paper for more details.

<br/>

 ### **Late concat multiple FC layers (patch and noisy patch)**:

This network is implemented in the follwing code :

* See the file for training :  Combined_Patch_Word/Font_Recognition_Multiple_5/**train_network_weighted_concat_equal.py**

* See the file for network architecture "**CombineMultiOutputModelWeightedConcat1**" :  Combined_Patch_Word/Font_Recognition_Multiple_5/**network_model.py**

* See the file for actual training using loop and epochs :  Combined_Patch_Word/Font_Recognition_Multiple_5/**trainer_weighted_concat.py**

* In this case, you need to use **"multi_image_data_loader_new_ver1.py"** file instead of **"multi_image_data_loader_new.py"** file in line 10 of **"train_network_weighted_concat_equal.py"**

The results can be seen in the Table.S9, 1<sup>st</sup> row of the paper. Please see the paper for more details. 

<br/>

 ### **Late concat AlexNet like Conv. layers (patch and noisy patch)**:

This network is implemented in the follwing code :

* See the file for training :  Combined_Patch_Word/Font_Recognition_Multiple_5/**train_network_weighted_concat_equal_alexnet.py**

* See the file for network architecture "**CombineMultiOutputModelWeightedConcatAlexNet**" :  Combined_Patch_Word/Font_Recognition_Multiple_5/**network_model.py**

* See the file for actual training using loop and epochs :  Combined_Patch_Word/Font_Recognition_Multiple_5/**trainer_weighted_concat_alexnet.py**

* In this case, you need to use **"multi_image_data_loader_new_ver1.py"** file instead of **"multi_image_data_loader_new.py"** file in line 10 of **"train_network_weighted_concat_equal_alexnet.py"**

The results can be seen in the Table.S9, 2<sup>nd</sup> row of the paper. Please see the paper for more details.

<br/>

 ### **Late concat multiple FC layers (without softmax layers)**:

This network is implemented in the follwing code :

* See the file for training :  Combined_Patch_Word/Font_Recognition_Multiple_5/**train_network_weighted_concat_equal_softmax.py**

* See the file for network architecture "**CombineMultiOutputModelWeightedConcat2**" :  Combined_Patch_Word/Font_Recognition_Multiple_5/**network_model.py**

* See the file for actual training using loop and epochs :  Combined_Patch_Word/Font_Recognition_Multiple_5/**trainer_weighted_concat.py**


The results can be seen in the Table.S10, 1<sup>st</sup> row of the paper. Please see the paper for more details. 

<br/>

 ### **Late concat AlexNet like Conv. layers (without softmax layers)**:

This network is implemented in the follwing code :

* See the file for training :  Combined_Patch_Word/Font_Recognition_Multiple_5/**train_network_weighted_concat_equal_alexnet_softmax.py**

* See the file for network architecture "**CombineMultiOutputModelWeightedConcatAlexNet_1**" :  Combined_Patch_Word/Font_Recognition_Multiple_5/**network_model.py**

* See the file for actual training using loop and epochs :  Combined_Patch_Word/Font_Recognition_Multiple_5/**trainer_weighted_concat_alexnet_softmax.py**


The results can be seen in the Table.S10, 2<sup>nd</sup> row of the paper. Please see the paper for more details.

<br/>

<br/>



## Testing of MTL+MI network and Weighted MTL+MI network
The following code is to test MTL+MI network and Weighted MTL+MI network. We have performed the operation by using the word images and patch images together. 

* See the file for testing :  Combined_Patch_Word/Font_Recognition_Multiple_5/**test_patch_multiple_all.py**

* You just need to modify the path of saved model path in line 21
* Import the correct network model in line 9
* Change the line 52 according to the correct model
* That's all ! 



<br/>
<br/>

## Some more Ablation Study
Furthermore, we did some more Ablation study/experiments also. These studies are mentioend below :

<br/>

###  **MTL Word: Word Level with multiple FC layers (on small dataset)** :

Here, we are checking whether the creation of dataloader is correct or not. To do this, we perform one kind of cross verification. We use the same dataloader creation code which was written to read the word and patch images together. This dataloader code was used in **MTL + MI** based network. But here, during the training, we are using only the "word" images. We want to see whether it is giving the same results like when we use the dataloader code to read either word or patch images only (not the both). 

Please note that this network architecture can only accept/take single image as the input (it can't take two input)

* See the file for training :  Multi_Tasks/Small_Dataset_Word_Level/Ablation_Study/Font_Recognition_Multiple_Test_1/**train_network_patch.py**

* Have a look at the code: Multi_Tasks/Small_Dataset_Word_Level/Ablation_Study/Font_Recognition_Multiple_Test_1/ **"word_image_datasets.py"** and see how the dataloader is written.

* Have a look at the code: Multi_Tasks/Small_Dataset_Word_Level/Ablation_Study/Font_Recognition_Multiple_Test_1/**"trainer.py"** and see that we are using only "inputs_2" in line 121 i.e. we are giving only word as input

<br/>


###  **MTL Patch: Patch Level with multiple FC layers (on small dataset)** :

In the simillar way, we only experiment with patch images here

* See the file for training :  Multi_Tasks/Small_Dataset_Patch_Level/Ablation_Study/Font_Recognition_Multiple_Test_2/**train_network_patch.py**

* Have a look at the code: Multi_Tasks/Small_Dataset_Patch_Level/Ablation_Study/Font_Recognition_Multiple_Test_2/ **"word_image_datasets.py"** and see how the dataloader is written.

* Have a look at the code: Multi_Tasks/Small_Dataset_Patch_Level/Ablation_Study/Font_Recognition_Multiple_Test_2/**"trainer.py"** and see that we are using only "inputs_2" in line 121 i.e. we are giving only patch as input

<br/>

###  **MTL+MI : Late concat multiple FC layers (on small dataset)** :
The objective of this experiment was mainly to verify the cause of network overfitting. As we are using word and patch images together in this network. So, here we try to understand among word and patch images, which one is main culprit for overfitting i.e. either word or patch images


**Giving same word images as both the input to the network**

* See the file for training :  Combined_Patch_Word/Ablation_Study/Font_Recognition_Multiple_1/**train_network_concat_equal.py**

* Have a look at the code: Combined_Patch_Word/Ablation_Study/Font_Recognition_Multiple_1/ **"multi_image_data_loader_new.py"** and see how the dataloader is written.

* Have a look at the code: Combined_Patch_Word/Ablation_Study/Font_Recognition_Multiple_1/**"trainer.py"** and see that we are using only "inputs_1" in line 134 i.e. we are giving only word image as both the inputs


**Giving same patch images as both the input to the network**

* See the file for training :  Combined_Patch_Word/Ablation_Study/Font_Recognition_Multiple_2/**train_network_concat_equal.py**

* Have a look at the code: Combined_Patch_Word/Ablation_Study/Font_Recognition_Multiple_2/ **"multi_image_data_loader_new.py"** and see how the dataloader is written.

* Have a look at the code: Combined_Patch_Word/Ablation_Study/Font_Recognition_Multiple_2/**"trainer.py"** and see that we are using only "inputs_2" in line 134 i.e. we are giving only patch image as both the inputs


**Giving word and patch images as the input to the network but different dataloader code**

Here, we have used the word and patch images as the input to the network. But, we tried different way of writing the dataloader so verify whether there are any issue with the dataloader. 

* See the file for training :  Combined_Patch_Word/Ablation_Study/Font_Recognition_Multiple_3/**train_network_concat_equal.py**

* Have a look at the code: Combined_Patch_Word/Ablation_Study/Font_Recognition_Multiple_3/ **"multi_image_data_loader_new.py"** and see how the dataloader is written.

* Have a look at the code: Combined_Patch_Word/Ablation_Study/Font_Recognition_Multiple_3/**"trainer.py"** and see that we are using "inputs_1"  in line 134 i.e. we are giving only word image as both the inputs

<br/>
<br/>

## Drawing the Plots 
In the following section, we have mentioend the details of the code, used to draw the plots which are mentioend in the paper. The code is in MatLab

* See the file for plotting the graph :  
  * /Font_Recognition/Plots_Graphs/**read_plot_Graph_1.m** : Here we are plotting the training and validation accuracies of word and patch images together  
    * The plot is shown in Figure. 6a and 6b

  * /Font_Recognition/Plots_Graphs/**read_plot_Graph_2.m** : Here we are plotting only the training accuracies of word and patch images together
     * The plot is shown in Figure. 5


## Full page level accuracy 
The full page level accuracies are mentioend in Table. 4 in the paper

* Full page level accuracies using segmented words :  
  * /Font_Recognition/Multi_Tasks/Word_Level/**test_word_page_level.py** 

* Full page level accuracies using segmented patches :  
  * /Font_Recognition/Multi_Tasks/Patch_Level/**test_word_page_level.py** 