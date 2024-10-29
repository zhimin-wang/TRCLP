import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline      # for warping
from transforms3d.axangles import axangle2mat  # for rotation
#from transforms2d.axangles import axangle2mat  # for rotation

class augmentations(object):
    def __init__(self):
        self.jitter_scale_ratio = 1.1
        self.jitter_ratio = 0.8

def probability():
    return np.random.rand()

def DataTransform(sample, config):

    weak_aug = scaling(sample, config.augmentation.jitter_scale_ratio)
    strong_aug = jitter(permutation(sample, max_segments=config.augmentation.max_seg), config.augmentation.jitter_ratio)
    
    return weak_aug, strong_aug


def flip_input_output(x, label):

    p = probability()
    if p < 0.5:
        x = data_flip(x)
        label_new = torch.flip(label, dims=[0])
        return x.type(torch.FloatTensor), label_new
    else:
        return x, label


def only_scale(x):
    x = scale_data(x)  
    return x.type(torch.FloatTensor)

def sub_flip(x):
    x1 = np.array(x)
    len = x1.shape[-1] // 2
    x1_column = x1.reshape(len, 2)
    x1_reverse = x1_column[::-1]
    x1_reverse = x1_reverse.reshape(-1, len * 2)[0]
    return torch.from_numpy(x1_reverse)

def data_flip(x):
    x1, x2, x3, len = data_split(x)
    # print(x1)
    x1_reverse = sub_flip(x1)
    # print(x1_reverse)
    x2_reverse = sub_flip(x2)
    x3_reverse = sub_flip(x3)
    x_tmp = torch.cat((x1_reverse, x2_reverse), 0)
    x_tmp2 = torch.cat((x_tmp, x3_reverse), 0)
    return x_tmp2

def flip_aug(x):
    x = data_flip(x)
    return x.type(torch.FloatTensor)

def scale_aug(x):
    x = scale_data(x)
    return x.type(torch.FloatTensor)


def y_flip(y):
    y = np.array(y)
    y_reverse = y[::-1]
    y_reverse = y_reverse.copy()
    return torch.from_numpy(y_reverse)

def dataAugmentation(x):

    p = probability()
    if p < 0.5:
        x = data_flip(x)     
    else:
        x = scale_data(x)

    
    return x.type(torch.FloatTensor)


def data_split(x):
    len = x.shape[-1] // 3
    x1 = x[0:len]
    x2 = x[len:len*2]
    x3 = x[len*2:len*3]
    return x1, x2, x3, len

num = 0

## This example using cubic splice is not the best approach to generate random curves. 
## You can use other aprroaches, e.g., Gaussian process regression, Bezier curve, etc.
def GenerateRandomCurves(X, sigma=0.2, knot=4):
    xx = (np.ones((1,1))*(np.arange(0,X.shape[0], (X.shape[0]-1)/(knot+1)))).transpose()
    yy = np.random.normal(loc=1.0, scale=sigma, size=(knot+2, 1))
    x_range = np.arange(X.shape[0])
    cs_x = CubicSpline(xx[:,0], yy[:,0])
    return torch.from_numpy(np.array(cs_x(x_range)))

def GenerateRandomCurves2(X, sigma=0.2, knot=4):
    xx = (np.ones((X.shape[1],1))*(np.arange(0,X.shape[0], (X.shape[0]-1)/(knot+1)))).transpose()
    yy = np.random.normal(loc=1.0, scale=sigma, size=(knot+2, X.shape[1]))
    x_range = np.arange(X.shape[0])
    cs_x = CubicSpline(xx[:,0], yy[:,0])
    cs_y = CubicSpline(xx[:,1], yy[:,1])
    return np.array([cs_x(x_range),cs_y(x_range)]).transpose()

def DA_MagWarp(X, sigma, knot):
    return X * GenerateRandomCurves(X, sigma, knot)

def magWarping(x):
    x1, x2, x3, len = data_split(x)

    sigma = 0.08
    knot = 4
    #global num
    # print("---------------")
    x11 = DA_MagWarp(x1, sigma, knot)
    x22 = DA_MagWarp(x2, sigma, knot)
    x33 = DA_MagWarp(x3, sigma, knot)

    # visualize = True
    # plt.cla()
    # plt.clf()
    # num += 1
    # if visualize and num % 50 == 0:
    #     fig = plt.figure(figsize=(10,8))

    #     ax = fig.add_subplot(3,2,1)
    #     x1_two_column = x1.reshape(50, 2)
    #     ax.plot(x1_two_column)
    #     #ax.set_title(str(factor1))
    #     ax.set_xlim([0,50])
    #     ax.set_ylim([0,1])

    #     ax = fig.add_subplot(3,2,2)
    #     x11_two_column = x11.reshape(50, 2)
    #     ax.plot(x11_two_column)
    #     ax.set_xlim([0,50])
    #     ax.set_ylim([0,1])

    #     ax = fig.add_subplot(3,2,3)
    #     x2_two_column = x2.reshape(50, 2)
    #     ax.plot(x2_two_column)
    #     #ax.set_title(str(factor2))
    #     ax.set_xlim([0,50])
    #     ax.set_ylim([-50,50])

    #     ax = fig.add_subplot(3,2,4)
    #     x22_two_column = x22.reshape(50, 2)
    #     ax.plot(x22_two_column)
    #     ax.set_xlim([0,50])
    #     ax.set_ylim([-50,50])

    #     ax = fig.add_subplot(3,2,5)
    #     x3_two_column = x3.reshape(50, 2)
    #     #ax.set_title(str(factor3))
    #     ax.plot(x3_two_column)
    #     ax.set_xlim([0,50])
    #     ax.set_ylim([-50,50])

    #     ax = fig.add_subplot(3,2,6)
    #     x33_two_column = x33.reshape(50, 2)
    #     ax.plot(x33_two_column)
    #     ax.set_xlim([0,50])
    #     ax.set_ylim([-50,50])

    #     plt.savefig('../utils/testdata/plot' + str(num) + '.jpg')

    x_tmp = torch.cat((x11, x22), 0)
    x_tmp2 = torch.cat((x_tmp, x33), 0)

    return x_tmp2

def DistortTimesteps(X, sigma=0.2):
    tt = GenerateRandomCurves2(X, sigma) # Regard these samples aroun 1 as time intervals
    # print(X.shape, tt.shape)
    tt_cum = np.cumsum(tt, axis=0)        # Add intervals to make a cumulative graph
    # Make the last value to have X.shape[0]
    
    t_scale = [(X.shape[0]-1)/tt_cum[-1,0],(X.shape[0]-1)/tt_cum[-1,1]]
    tt_cum[:,0] = tt_cum[:,0]*t_scale[0]
    tt_cum[:,1] = tt_cum[:,1]*t_scale[1]
    return tt_cum

def DA_TimeWarp(X, sigma=0.2):
    tt_new = DistortTimesteps(X.numpy(), sigma)
    X_new = np.zeros(X.shape)
    x_range = np.arange(X.shape[0])
    X_new[:,0] = np.interp(x_range, tt_new[:,0], X[:,0])
    X_new[:,1] = np.interp(x_range, tt_new[:,1], X[:,1])
    return torch.from_numpy(X_new)

def timeWarping(x):
    x1, x2, x3, len = data_split(x)
    half_len = len // 2
    sigma = 0.15
    # global num
    # print("---------------")
    # x11 = DA_TimeWarp(x1, sigma, knot)
    # x22 = DA_TimeWarp(x2, sigma, knot)
    # x33 = DA_TimeWarp(x3, sigma, knot)

    x11 = DA_TimeWarp(x1.reshape(half_len, 2), sigma)
    x22 = DA_TimeWarp(x2.reshape(half_len, 2), sigma)
    x33 = DA_TimeWarp(x3.reshape(half_len, 2), sigma)

    x11 = x11.reshape(len)
    x22 = x22.reshape(len)
    x33 = x33.reshape(len)

    # visualize = True
    # plt.cla()
    # plt.clf()
    # num += 1
    # if visualize and num % 50 == 0:
    #     fig = plt.figure(figsize=(10,8))

    #     ax = fig.add_subplot(3,2,1)
    #     x1_two_column = x1.reshape(50, 2)
    #     ax.plot(x1_two_column)
    #     #ax.set_title(str(factor1))
    #     ax.set_xlim([0,50])
    #     ax.set_ylim([0,1])

    #     ax = fig.add_subplot(3,2,2)
    #     x11_two_column = x11.reshape(50, 2)
    #     ax.plot(x11_two_column)
    #     ax.set_xlim([0,50])
    #     ax.set_ylim([0,1])

    #     ax = fig.add_subplot(3,2,3)
    #     x2_two_column = x2.reshape(50, 2)
    #     ax.plot(x2_two_column)
    #     #ax.set_title(str(factor2))
    #     ax.set_xlim([0,50])
    #     ax.set_ylim([-50,50])

    #     ax = fig.add_subplot(3,2,4)
    #     x22_two_column = x22.reshape(50, 2)
    #     ax.plot(x22_two_column)
    #     ax.set_xlim([0,50])
    #     ax.set_ylim([-50,50])

    #     ax = fig.add_subplot(3,2,5)
    #     x3_two_column = x3.reshape(50, 2)
    #     #ax.set_title(str(factor3))
    #     ax.plot(x3_two_column)
    #     ax.set_xlim([0,50])
    #     ax.set_ylim([-50,50])

    #     ax = fig.add_subplot(3,2,6)
    #     x33_two_column = x33.reshape(50, 2)
    #     ax.plot(x33_two_column)
    #     ax.set_xlim([0,50])
    #     ax.set_ylim([-50,50])

    #     plt.savefig('../utils/testdata/plot' + str(num) + '.jpg')

    x_tmp = torch.cat((x11, x22), 0)
    x_tmp2 = torch.cat((x_tmp, x33), 0)

    return x_tmp2




def jitter(x, sigma=0.8):
    # https://arxiv.org/pdf/1706.00527.pdf

    global num
    # split data into three parts due to data range
    x1, x2, x3, len = data_split(x)

    sigma1 = 0.05
    sigma2 = 0.8
    sigma3 = 0.8
    
    visualize = True

    
    
    x11 = x1 + np.random.normal(loc=0., scale=sigma1, size=x1.shape)
    x22 = x2 + np.random.normal(loc=0., scale=sigma2, size=x2.shape)
    x33 = x3 + np.random.normal(loc=0., scale=sigma3, size=x3.shape)
    plt.cla()
    plt.clf()
    num += 1
    if visualize and num % 50 == 0:
        fig = plt.figure(figsize=(15,4))

        ax = fig.add_subplot(2,3,1)
        x1_two_column = x1.reshape(50, 2)
        ax.plot(x1_two_column)
        ax.set_xlim([0,50])
        ax.set_ylim([0,1])

        ax = fig.add_subplot(2,3,4)
        x11_two_column = x11.reshape(50, 2)
        ax.plot(x11_two_column)
        ax.set_xlim([0,50])
        ax.set_ylim([0,1])

        ax = fig.add_subplot(2,3,2)
        x2_two_column = x2.reshape(50, 2)
        ax.plot(x2_two_column)
        ax.set_xlim([0,50])
        ax.set_ylim([-50,50])

        ax = fig.add_subplot(2,3,5)
        x22_two_column = x22.reshape(50, 2)
        ax.plot(x22_two_column)
        ax.set_xlim([0,50])
        ax.set_ylim([-50,50])

        ax = fig.add_subplot(2,3,3)
        x3_two_column = x3.reshape(50, 2)
        ax.plot(x3_two_column)
        ax.set_xlim([0,50])
        ax.set_ylim([-50,50])

        ax = fig.add_subplot(2,3,6)
        x33_two_column = x33.reshape(50, 2)
        ax.plot(x33_two_column)
        ax.set_xlim([0,50])
        ax.set_ylim([-50,50])

        plt.savefig('../utils/testdata/plot' + str(num) + '.jpg')
        

    x_tmp = torch.cat((x11, x22), 0)
    x_tmp2 = torch.cat((x_tmp, x33), 0)

    return x_tmp2

def scaling(x, sigma=0.1):
    # https://arxiv.org/pdf/1706.00527.pdf
    factor = np.random.normal(loc=1., scale=sigma, size=(1,1))
    myNoise = np.matmul(np.ones((x.shape[0], 1)), factor)
    # print("---factor:", factor)
    # print("---x:", x)
    myNoise = np.squeeze(myNoise)
    x = x * myNoise
    # print("---x:", x)
    # print("-------------:", factor[0][0])
    return factor[0][0],x
    #  return x * myNoise

def scale_data(x):
    x1, x2, x3, len = data_split(x)
    # global num
    sigma = 0.15
    # print("---------------")
    factor1,x11 = scaling(x1, sigma)
    factor2,x22 = scaling(x2, sigma)
    factor3,x33 = scaling(x3, sigma)
    # visualize = True
    # plt.cla()
    # plt.clf()
    # num += 1
    # if visualize and num % 50 == 0:
        # fig = plt.figure(figsize=(10,8))

        # ax = fig.add_subplot(3,2,1)
        # x1_two_column = x1.reshape(50, 2)
        # ax.plot(x1_two_column)
        # ax.set_title(str(factor1))
        # ax.set_xlim([0,50])
        # ax.set_ylim([0,1])

        # ax = fig.add_subplot(3,2,2)
        # x11_two_column = x11.reshape(50, 2)
        # ax.plot(x11_two_column)
        # ax.set_xlim([0,50])
        # ax.set_ylim([0,1])

        # ax = fig.add_subplot(3,2,3)
        # x2_two_column = x2.reshape(50, 2)
        # ax.plot(x2_two_column)
        # ax.set_title(str(factor2))
        # ax.set_xlim([0,50])
        # ax.set_ylim([-50,50])

        # ax = fig.add_subplot(3,2,4)
        # x22_two_column = x22.reshape(50, 2)
        # ax.plot(x22_two_column)
        # ax.set_xlim([0,50])
        # ax.set_ylim([-50,50])

        # ax = fig.add_subplot(3,2,5)
        # x3_two_column = x3.reshape(50, 2)
        # ax.set_title(str(factor3))
        # ax.plot(x3_two_column)
        # ax.set_xlim([0,50])
        # ax.set_ylim([-50,50])

        # ax = fig.add_subplot(3,2,6)
        # x33_two_column = x33.reshape(50, 2)
        # ax.plot(x33_two_column)
        # ax.set_xlim([0,50])
        # ax.set_ylim([-50,50])

        # plt.savefig('../utils/testdata/plot' + str(num) + '.jpg')

    x_tmp = torch.cat((x11, x22), 0)
    x_tmp2 = torch.cat((x_tmp, x33), 0)

    return x_tmp2


def permutation(x, max_segments=5, seg_mode="random"):
    orig_steps = np.arange(x.shape[2])

    num_segs = np.random.randint(1, max_segments, size=(x.shape[0]))

    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        if num_segs[i] > 1:
            if seg_mode == "random":
                split_points = np.random.choice(x.shape[2] - 2, num_segs[i] - 1, replace=False)
                split_points.sort()
                splits = np.split(orig_steps, split_points)
            else:
                splits = np.array_split(orig_steps, num_segs[i])
            warp = np.concatenate(np.random.permutation(splits)).ravel()
            ret[i] = pat[0,warp]
        else:
            ret[i] = pat
    return torch.from_numpy(ret)
