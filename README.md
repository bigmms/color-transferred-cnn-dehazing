# Color-Transferred-CNN-Dehazing
## *Color Transferred Convolutional Neural Networks for Image Dehazing*
Jia-Li Yin, Yi-Chi Huang, Bo-Hao Chen, and Shao-Zhen Ye

![](/demo.png)

## Prerequisites:
* Linux
* Python 2.7
* Numpy 1.14.0
* OpenCV 3.4.0.12
* Keras 2.1.5

## It was tested and runs under the following OSs:
* Ubuntu 18.04
* Elementary OS
Might work under others, but didn't get to test any other OSs just yet.

## Getting Started:
### Installation
- Install python libraries and requests.
```bash
pip install -r requirements.txt
```

### Testing 
- To test the model:
```bash
python demo.py
``` 
The test results will be saved in: `./TestOutput/.`

## Citation:
    @ARTICLE{yin2019dehazing, 
    author={J. {Yin} and Y. {Huang} and B. {Chen} and S. {Ye}}, 
    journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
    title={Color Transferred Convolutional Neural Networks for Image Dehazing}, 
    year={2019}, 
    volume={}, 
    number={}, 
    pages={1-1}, 
    keywords={Atmospheric modeling;Image color analysis;Integrated circuit modeling;Estimation;Standards;Computational modeling;Image dehazing;color transfer;deep learning}, 
    doi={10.1109/TCSVT.2019.2917315}, 
    ISSN={1051-8215}, 
    month={},}
