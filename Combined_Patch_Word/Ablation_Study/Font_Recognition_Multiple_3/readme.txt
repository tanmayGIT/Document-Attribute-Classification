this code is for multiple task of word images. Here I am trying on small dataset that means taking only few (300) images from all the different combination of possible classes. The same is done for the case of validation that means by taking only few (60) images from each classes. 

To select the few images from each classes, I choose them based on the amount of foreground pixels available in the image. 
This is the code where, I can only read two images in the data loader. Hence, I only train by using these two image. But here in this code, I give both the word and patch images. here I have rewritten the code of data loader to see whether the previus one may have problems
