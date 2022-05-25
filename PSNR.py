import os
import sys
from PIL import Image
import numpy as np
import copy

# ITU-R BT.601
# https://en.wikipedia.org/wiki/YCbCr
# RGB -> YCbCr
def rgb2ycbcr(rgb):
    m = np.array([[ 65.481, 128.553, 24.966],
                  [-37.797, -74.203, 112],
                  [ 112, -93.786, -18.214]])
    shape = rgb.shape
    if len(shape) == 3:
        rgb = rgb.reshape((shape[0] * shape[1], 3))
    ycbcr = np.dot(rgb, m.transpose() / 255.)
    ycbcr[:,0] += 16.
    ycbcr[:,1:] += 128.
    return ycbcr.reshape(shape)

# ITU-R BT.601
# https://en.wikipedia.org/wiki/YCbCr
# YUV -> RGB
def ycbcr2rgb(ycbcr):
    m = np.array([[ 65.481, 128.553, 24.966],
                  [-37.797, -74.203, 112],
                  [ 112, -93.786, -18.214]])
    shape = ycbcr.shape
    if len(shape) == 3:
        ycbcr = ycbcr.reshape((shape[0] * shape[1], 3))
    rgb = copy.deepcopy(ycbcr)
    rgb[:,0] -= 16.
    rgb[:,1:] -= 128.
    rgb = np.dot(rgb, np.linalg.inv(m.transpose()) * 255.)
    return rgb.clip(0, 255).reshape(shape)

def PSNRSingleChannel(x, y, max = 255) :
    assert (x.shape == y.shape), "Size mismatch."
    epi = 0.0001
    #print("np.product(x.shape)",np.product(x.shape))
    mse = np.sum(np.power((x - y), 2)) / (np.product(x.shape)) + epi
    #print("mse",mse)
    return 10 * np.log(max * max / mse) / np.log(10)

def usageHalt() :
    print("Usage: PSNR.py folderOfGT folderOfOutput")
    exit()
#
arg = sys.argv
#
# if (len(arg) != 3) :
#     usageHalt()


# if (not os.path.isdir(path1)) :
#     print(arg[1], "(extended as", path1, ")", 'is not a valid folder.')
#     usageHalt()
#
# if (not os.path.isdir(path2)) :
#     print(arg[2], "(extended as", path2, ")", 'is not a valid folder.')
#     usageHalt()

def get_PSNR(classname):
    if arg[1] == 'base':
        # 直接生成SR图片的base line 结果
        path1 = "/home/yuzhongkai/super_resolution/min_data/SR_result/" + classname  # SR result
        path2 = "/home/yuzhongkai/super_resolution/min_data/GT/" + classname  # GT
    elif arg[1] == 'new':
        path1 = "/home/yuzhongkai/super_resolution/function2/min_test/remap_result/" + classname   # folder of output
        path2 = "/home/yuzhongkai/super_resolution/min_data/GT/" + classname  # GT

    else:
        num = arg[1]
        path1 = "/home/yuzhongkai/super_resolution/function2/test/test" + num+'/'+classname+'/'  # folder of output
        path2 = "/home/yuzhongkai/super_resolution/Vid4/GT_bf/" + classname  # folder of ground truth




    # path2 = path1
    list1 = list(filter(os.path.isfile,map(lambda x: os.path.join(path1, x), os.listdir(path1))))
    list2 = list(filter(os.path.isfile,map(lambda x: os.path.join(path2, x), os.listdir(path2))))

    list1.sort()
    list2.sort()
    # print(list1)

    if (len(list1) != len(list2)) :
        print('Numbers of files contained in two folder is different.', (len(list1), len(list2)))
        if (len(list1) > len(list2)) :
            print('Too few image in target folder. Halt.')
            exit()

    listAns = []

    def arrayYielder(arr) :
        for i in range(arr.shape[-1]) :
            yield arr[..., i]

    for p1,p2 in zip(list1, list2) :
        # print('hi')
        img1 = Image.open(p1)
        # arr1 = rgb2ycbcr(np.array(img1, dtype = "double"))
        arr1 = (np.array(img1, dtype = "double"))
        img2 = Image.open(p2).resize(img1.size, Image.BICUBIC)
        # arr2 = rgb2ycbcr(np.array(img2, dtype = "double"))
        arr2 = (np.array(img2, dtype = "double"))
        listAns += [np.sum(list(map(lambda pair: PSNRSingleChannel(pair[0], pair[1]), zip(arrayYielder(arr1),arrayYielder(arr2))))) / 3]

    avg = np.average(listAns)

    #print("Avg PSNR =", avg, "for", len(list1), "images in", arg[1], "and", arg[2], "( Std =", np.std(listAns), ")")
    print('classname: ', classname, end='')
    print(listAns)
    print('    avg: ', avg)
    return

def quick_PSNR(path1, path2):
    list1 = list(filter(os.path.isfile,map(lambda x: os.path.join(path1, x), os.listdir(path1))))
    list2 = list(filter(os.path.isfile,map(lambda x: os.path.join(path2, x), os.listdir(path2))))

    list1.sort()
    list2.sort()
    # print(list1)

    if (len(list1) != len(list2)) :
        print('Numbers of files contained in two folder is different.', (len(list1), len(list2)))
        if (len(list1) > len(list2)) :
            print('Too few image in target folder. Halt.')
            exit()

    listAns = []

    def arrayYielder(arr) :
        for i in range(arr.shape[-1]) :
            yield arr[..., i]

    for p1,p2 in zip(list1, list2) :
        # print('hi')
        img1 = Image.open(p1)
        # arr1 = rgb2ycbcr(np.array(img1, dtype = "double"))
        arr1 = (np.array(img1, dtype = "double"))
        img2 = Image.open(p2).resize(img1.size, Image.BICUBIC)
        # arr2 = rgb2ycbcr(np.array(img2, dtype = "double"))
        arr2 = (np.array(img2, dtype = "double"))
        listAns += [np.sum(list(map(lambda pair: PSNRSingleChannel(pair[0], pair[1]), zip(arrayYielder(arr1),arrayYielder(arr2))))) / 3]

    avg = np.average(listAns)
    return avg

#
# classes = ['calendar', 'city', 'foliage', 'walk']
if __name__ == '__main__':
    classes = ['walk']
    for classname in classes:
        get_PSNR(classname)
    # get_PSNR('calendar')
