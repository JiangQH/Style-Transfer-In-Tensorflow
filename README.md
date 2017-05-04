# Introduction
This repo contains two copies of code, both do the style transfer job. Using this code you can transfer styles from
any image to another. 

# The optimize method
It is a pycaffe implementation of the paper [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576), the paper provides a way of combine two images to one. While holding the original image's content, we can transfer the style from another image to this one. The image to transfer style from is called style_image, and the image to be transfered is called content image.  

## optimize results
![styled_image](./result/optimize/styled.jpg)  

## How to
**requirements**  
[caffe](http://caffe.berkeleyvision.org/) with python, and make sure the PYTHONPATH is correctly set.     
[pre-trained vgg16 model](https://gist.github.com/jimmie33/27c1c0a7736ba66c2395)    
[scipy&numpy](https://www.scipy.org/)   

**usage**   

``python optimize.py -c /pathto/content_img -s /pathto/style_img``  

You can see the code in optimize for more detail. Note since it has to optimize, several minutes can be taken to complete the whole, and time to consume is related to the image size.  


# The convolutional way - fast style-transfer
Actually the optimization can be replaced by training a feedforward CNNs. Doing this when deploy, we can achieve real time style-transfer. Ideas from the following papers:    

[Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155)   

[Instance Normalization: The Missing Ingredient for Fast Stylization](https://arxiv.org/abs/1607.08022)  

And when doing this project, I find caffe's way to add new layer can be really complicated and not friendly to python interface. So I change the platform to tensorflow. 

## fast transfer results
It will be release soon with video transfered result.

## How to
**requirements**
[tensorflow](https://www.tensorflow.org/)  

[scipy&numpy](https://www.scipy.org/)  

**Train from scratch**
Before you train the network, some extra requirements are needed  
[COCO dataset](http://msvocds.blob.core.windows.net/coco2014/train2014.zip), release it in a content folder  

[pre-trained vgg16 model](), note this model has been converted to be loaded with numpy interface  

[pyyaml](http://pyyaml.org/), this is needed to parse the config file  

Now you can train your own style network, change the contents in conf.yml with guidance in it, everything is done. 

``python train.py -c conf.yml`` 

**Deploy a trained model**
Use the fast_generate script as follows: 

``python fast_generate.py -m <path to the trained model dir> -i <folders contain image want to transfer> -o <output dir>``  

My own pretrained models will be released soon, coming with some extra codes that can deploy the model to videos and cameras, so we can do real time transfer.
