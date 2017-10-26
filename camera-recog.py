#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @brief image/camera input recognition using pre-trained models
# @section Requirements:  python3,  chainer 2, OpenCV 3
# @version 0.01
# @date Aug. 2017
# @author Shizuo KAJI (shizuo.kaji@gmail.com)
# @licence MIT

import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
import threading
import PIL.Image as Image
import cv2
import argparse

# list of pre-trained models in chainer
archs = {
	'googlenet': L.GoogLeNet,
	'resnet50': L.ResNet50Layers,
	'resnet101': L.ResNet101Layers,
	'resnet152': L.ResNet152Layers,
	'vgg': L.VGG16Layers,
}

# command-line argument parsing
parser = argparse.ArgumentParser(description='Image recognition')
parser.add_argument('--input', '-i', 
                        help='text file containing a list of image filen paths')
parser.add_argument('--root', '-R', default=".", 
                        help='root dir for image files')
parser.add_argument('--arch', '-a', choices=archs.keys(), default='googlenet',
					help='Convnet architecture')
parser.add_argument('--num', '-n', type=int, default=5,
					help='number of guesses')
parser.add_argument('--gpu', '-g', type=int, default=-1,
					help='GPU ID (negative value indicates CPU')
args = parser.parse_args()

# set NN
chainer.config.train = False
func = archs[args.arch]()

# select GPU
if args.gpu >= 0:
	chainer.cuda.get_device_from_id(args.gpu).use()
	func.to_gpu()
xp = chainer.cuda.cupy if args.gpu >= 0 else np
	
# load synset labels; discard first 10 letters indicating ID
synset = np.array([line[10:-1] for line in open("synset.txt", 'r')])

# classifier
def predict(images,args):
    global synset
    prediction = func.predict(images)
    if args.gpu >= 0:
        pred = prediction.data.get()
    else:
        pred = prediction.data
    result = zip(pred.reshape((pred.size,)), synset)
    result = sorted(result, reverse=True)
    msg = ''.join(['{:>4} {}'.format(score,label)+'\n' for (score, label) in result[:args.num]])
    print(msg)
    return msg

# capture camera input
class ImageProcessing(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        threading.Thread.daemon = True
        self.capture = cv2.VideoCapture(0)
        if self.capture.isOpened() is False:
            raise("Camera IO Error")
        self.msg = ''
        self.pause = False
        ret, self.image = self.capture.read()

    def run(self):
        while True:
            if self.pause == True:
                continue
            img = Image.fromarray(self.image.copy())
            self.msg = predict([img],args)

if __name__ == "__main__":
    # when image files are given
    if args.input:
        with open(args.input) as input:
            for line in input:
                print("\n"+line)
                img = Image.open(args.root+"/"+line.strip())
                predict([img],args)
        exit(0)

    # camera input
    print("Hit 'q' to exit or 'p' to pause")
    windowname = "Recognition"
    cv2.namedWindow(windowname,cv2.WINDOW_AUTOSIZE)
    ip = ImageProcessing()
    ip.start()

    while True:
        ret, ip.image = ip.capture.read()
        # is image captured?
        if ret == False:
            continue
        ip.image = ip.image[:,::-1,:]
        cv2.imshow(windowname, ip.image)
        key = cv2.waitKey(50)
        if key & 0xFF == ord("q"):  # when q key is pressed
            ip.pause = True
            break
        elif key & 0xFF == ord("p"):
            ip.pause = not ip.pause
			
    ip.capture.release()
    cv2.destroyAllWindows()
