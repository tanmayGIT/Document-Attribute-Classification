this code is for multiple task of patch images. Here I am trying on small dataset that means taking only few (300) images from all the different combination of possible classes. The same is done for the case of validation that means by taking only few (60) images from each classes. 

To select the few images from each classes, I choose them based on the amount of foreground pixels available in the image. 

The trial here is to check whether we can get symmetrical results on only small dataset ?

Here the idea was to see when we do the combination of the patch and word images (by taking the same small number of images from the word folder or from the patch folder), what type of results, we get. 
We create the data loader by combining them both. Then during the training, we are using only patch images for training. We want to see whether it is giving the same results as we were getting when using a separate training dataset for patch only.

More important thing is that the network architecture here can't accept two inputs. It can only access one input (this network is different than the network which can acess two inputs). 
