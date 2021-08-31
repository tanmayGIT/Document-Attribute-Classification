this code is for multiple task of patch images. Here I am trying on small dataset that means taking only few (300) images from all the different combination of possible classes. The same is done for the case of validation that means by taking only few (60) images from each classes. 

To select the few images from each classes, I choose them based on the amount of foreground pixels available in the image. 

The trial here is to check whether we can get symmetrical results on only small dataset ?
The test here is done by reading only patch images and using the patch images while training (this network can only take one input).
