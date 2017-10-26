Image/camera input recognition using pre-trained models
=============
A python demo code for image/camera input recognition
 using pre-trained models in Chainer
by Shizuo KAJI

## Licence
MIT Licence

# Requirements
- python 3.6

Install [Anaconda](https://www.anaconda.com/download/)
 if you do not have python 3 on your system.

- Chainer > 2:  `pip install chainer`

If you'd like to use GPU, cupy with CUDA support is additionally required.

- OpenCV 3: Use Homebrew on mac, or download the binary from [here](http://www.lfd.uci.edu/%7Egohlke/pythonlibs/#opencv) and pip on Windows to install.

# Example
`python camera-recog.py -h`
for a brief description of command line arguments

`python camera-recog.py`
opens the camera of your PC and outputs recognised objects captured by it.
Hit 'p' to pause/resume, 'q' to quit.

`python camera-recog.py -i sample.txt -n 6`
scans the text file sample.txt and outputs 6 guesses for each image specified in the file.
Sample images are obtained from Wikipedia.

`python camera-recog.py -i sample.txt -n 5 -a vgg -g 0`
same as above but with the VGG model and GPU.
