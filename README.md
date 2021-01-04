# Color Transferred Convolutional Neural Networks for Image Dehazing
Jia-Li Yin, Yi-Chi Huang, Bo-Hao Chen, and Shao-Zhen Ye

[![HitCount](http://hits.dwyl.com/bigmms/bigmms/color-transferred-cnn-dehazing.svg)](http://hits.dwyl.com/bigmms/bigmms/color-transferred-cnn-dehazing)

![](/demo.png)

## Prerequisites:
* Linux
* Python 2.7
* Numpy 1.14.0
* OpenCV 3.4.0.12
* Keras 2.1.5
* Tensorflow 1.3.0
* Tflearn 0.3.2

## It was tested and runs under the following OSs:
* Ubuntu 18.04
* Elementary OS

Might work under others, but didn't get to test any other OSs just yet.

## Getting Started:
### Installation
- Install python libraries and requests.
```bash
pip install -r requirement.txt
```

### Testing 
- To test the model:
```bash
python demo.py
``` 
The test results will be saved in: `./TestOutput/.`

## Citation:
    @ARTICLE{yin2019dehazing,  
    author={J. -L. {Yin} and Y. -C. {Huang} and B. -H. {Chen} and S. -Z. {Ye}},  
    journal={IEEE Transactions on Circuits and Systems for Video Technology},  
    title={Color Transferred Convolutional Neural Networks for Image Dehazing},   
    year={2020},  
    volume={30},  
    number={11},  
    pages={3957-3967}, 
    doi={10.1109/TCSVT.2019.2917315}}
    
