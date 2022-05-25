import cv2
import numpy as np
import glob
from PIL import Image
# import openpyxl
# import xlwt
import csv
import math
import os
from progressbar import *
import re
import matplotlib.pyplot as plt

# IDX_DIR = "/home/songzhuoran/video/video-sr-acc/Vid4/Info_BIx4/idx/"
# PICS_DIR = "/home/yuzhongkai/super_resolution/min_data/BIx4/"
# SR_PICS_DIR = "/home/yuzhongkai/super_resolution/min_data/SR_result/"
# GT_PICS_DIR = "/home/yuzhongkai/super_resolution/min_data/GT/"
ORG_DATA_DIR="/home/yuzhongkai/E2SR/datasets/"
NEW_MVS_DIR = "remap_MV/"
# BACKUP_DIR = "/home/yuzhongkai/super_resolution/function2/min_test/remap_MV/backup/"
# Reconstruct_DIR = "/home/yuzhongkai/super_resolution/function2/min_test/remap_result/"
# Canny_DIR = '/home/yuzhongkai/super_resolution/function2/min_test/canny_result/'
# MV_FIG_DIR = '/home/yuzhongkai/super_resolution/function2/movtion_predict/result/'


def main_stat(classname):
    orgMvs = []
    newMvs = []
    bflist = []
    with open(IDX_DIR + "b/" + classname, "r") as file:
        for row in file:
            bflist.append(int(row) - 1)
        print('bflist: ', bflist, len(bflist))

    orgMvSize, newMvSize, orgResSize, newResSize = 0, 0, 0, 0
    block4, blockAll = 0, 0

    with open(MVS_DIR + classname + ".csv", "r") as file:
        reader = csv.reader(file)
        for item in reader:
            orgMvs.append(item)

    with open(OUT_DIR + "min_remap_cop_" + classname + ".csv", "r") as file:
        reader = csv.reader(file)
        for item in reader:
            newMvs.append(item)

    # print(len(orgMvs), len(newMvs))

    for mv in orgMvs:
        curId, refIf, blockw, blockh = np.array(mv[0:4]).astype(int)

        if curId not in bflist:
            continue
        orgMvSize += 7
        orgResSize += blockw * blockh * 3

    for mv in newMvs:
        curId, refIf, blockw, blockh = np.array(mv[0:4]).astype(int)
        if curId not in bflist:
            continue
        newMvSize += 7
        blockAll += 1
        if blockh == 4 and blockw == 4:
            newResSize += blockw * blockh * 3/16   # 4x4的块取完整残差
            block4 += 1
        else:
            newResSize += blockw/4 * blockh/4 * 3

    orgStreamSize = orgMvSize + orgResSize
    newStreamSize = newMvSize + newResSize

    print('4x4/all=%d/%d=%f' %(block4, blockAll, block4/blockAll))
    print('org res/mv=%d/%d=%d' %(orgResSize, orgMvSize, orgResSize/orgMvSize))
    print('new res/mv=%d/%d=%d' %(newResSize, newMvSize, newResSize/newMvSize))
    print('new/org=%d/%d=%f' %(newStreamSize, orgStreamSize, newStreamSize/orgStreamSize))

def quick_stat(dataset, classname, fullRes):
    orgMvs = []
    newMvs = []
    bflist = []

    IDX_DIR = "/home/yuzhongkai/E2SR/datasets/%s/Info_BIx4/idx/" % dataset
    with open(IDX_DIR + "b/" + classname, "r") as file:
        for row in file:
            bflist.append(int(row) - 1)
        # print('bflist: ', bflist, len(bflist))

    orgMvSize, newMvSize, orgResSize, newResSize = 0, 0, 0, 0
    block4, blockAll = 0, 0

    ORG_MVS_DIR = '%s%s/Info_BIx4/mvs/' % (ORG_DATA_DIR, dataset)
    with open(ORG_MVS_DIR + classname + "_loss.csv", "r") as file:
        reader = csv.reader(file)
        for item in reader:
            orgMvs.append(item)

    with open(NEW_MVS_DIR + "min_remap_cop_" + classname + ".csv", "r") as file:
        reader = csv.reader(file)
        for item in reader:
            newMvs.append(item)

    # print(len(orgMvs), len(newMvs))

    for mv in orgMvs:
        curId, refIf, blockw, blockh = np.array(mv[0:4]).astype(int)
        # if curId > 6:
        #     continue
        if curId not in bflist:
            continue
        orgMvSize += 7
        orgResSize += blockw * blockh * 3

    for mv in newMvs:
        curId, refIf, blockw, blockh = np.array(mv[0:4]).astype(int)
        if curId not in bflist:
            continue
        newMvSize += 7
        blockAll += 1
        if blockh == 4 and blockw == 4:
            if fullRes:
                newResSize += blockw * blockh * 3   # 4x4的块取完整残差
            else:
                newResSize += blockw/4 * blockh/4 * 3   # 4x4的块也压缩
            block4 += 1
        else:
            newResSize += blockw/4 * blockh/4 * 3

    orgStreamSize = orgMvSize + orgResSize
    newStreamSize = newMvSize + newResSize

    # print('4x4/all=%d/%d=%f' %(block4, blockAll, block4/blockAll))
    # print('org res/mv=%d/%d=%d' %(orgResSize, orgMvSize, orgResSize/orgMvSize))
    # print('new res/mv=%d/%d=%d' %(newResSize, newMvSize, newResSize/newMvSize))
    # print('new/org=%d/%d=%f' %(newStreamSize, orgStreamSize, newStreamSize/orgStreamSize))
    return newStreamSize/orgStreamSize

def quick_stat_class(bflist, org_bs, new_bs, fullRes=True):
    orgMvs = []
    newMvs = []

    orgMvSize, newMvSize, orgResSize, newResSize = 0, 0, 0, 0
    block4, blockAll = 0, 0

    with open(org_bs, "r") as file:
        reader = csv.reader(file)
        for item in reader:
            orgMvs.append(item)

    with open(new_bs, "r") as file:
        reader = csv.reader(file)
        for item in reader:
            newMvs.append(item)

    print(len(orgMvs), len(newMvs))

    for mv in orgMvs:
        curId, refIf, blockw, blockh = np.array(mv[0:4]).astype(int)
        # if curId > 6:
        #     continue
        if curId not in bflist:
            continue
        # print('orgMVsize', orgMvSize)
        orgMvSize += 7
        orgResSize += blockw * blockh * 3

    for mv in newMvs:
        curId, refIf, blockw, blockh = np.array(mv[0:4]).astype(int)
        if curId not in bflist:
            continue
        newMvSize += 7
        blockAll += 1
        if blockh == 4 and blockw == 4:
            if fullRes:
                newResSize += blockw * blockh * 3   # 4x4的块取完整残差
            else:
                newResSize += blockw/4 * blockh/4 * 3   # 4x4的块也压缩
            block4 += 1
        else:
            newResSize += blockw/4 * blockh/4 * 3

    # print(orgMvSize, orgResSize)
    # print(newMvSize, newResSize)
    orgStreamSize = orgMvSize + orgResSize
    newStreamSize = newMvSize + newResSize

    # print('4x4/all=%d/%d=%f' %(block4, blockAll, block4/blockAll))
    # print('org res/mv=%d/%d=%d' %(orgResSize, orgMvSize, orgResSize/orgMvSize))
    # print('new res/mv=%d/%d=%d' %(newResSize, newMvSize, newResSize/newMvSize))
    # print('new/org=%d/%d=%f' %(newStreamSize, orgStreamSize, newStreamSize/orgStreamSize))
    return newStreamSize/orgStreamSize

def quick_stat2(classname, fullRes, t1, t2):
    orgMvs = []
    newMvs = []
    bflist = []
    with open(IDX_DIR + "b/" + classname, "r") as file:
        for row in file:
            bflist.append(int(row) - 1)
        # print('bflist: ', bflist, len(bflist))

    orgMvSize, newMvSize, orgResSize, newResSize = 0, 0, 0, 0
    block4, blockAll = 0, 0

    with open(MVS_DIR + classname + ".csv", "r") as file:
        reader = csv.reader(file)
        for item in reader:
            orgMvs.append(item)


    with open(BACKUP_DIR + "remap_cop_%d_%d_%s.csv" % (t1, t2, classname), "r") as file:
        reader = csv.reader(file)
        for item in reader:
            newMvs.append(item)

    # print(len(orgMvs), len(newMvs))

    for mv in orgMvs:
        curId, refIf, blockw, blockh = np.array(mv[0:4]).astype(int)
        if curId > 6:
            continue
        if curId not in bflist:
            continue
        orgMvSize += 7
        orgResSize += blockw * blockh * 3

    for mv in newMvs:
        curId, refIf, blockw, blockh = np.array(mv[0:4]).astype(int)
        if curId not in bflist:
            continue
        newMvSize += 7
        blockAll += 1
        if blockh == 4 and blockw == 4:
            if fullRes:
                newResSize += blockw * blockh * 3   # 4x4的块取完整残差
            else:
                newResSize += blockw * blockh * 3 / 16  # 4x4的块也压缩
            block4 += 1
        else:
            newResSize += blockw/4 * blockh/4 * 3

    orgStreamSize = orgMvSize + orgResSize
    newStreamSize = newMvSize + newResSize

    # print('4x4/all=%d/%d=%f' %(block4, blockAll, block4/blockAll))
    # print('org res/mv=%d/%d=%d' %(orgResSize, orgMvSize, orgResSize/orgMvSize))
    # print('new res/mv=%d/%d=%d' %(newResSize, newMvSize, newResSize/newMvSize))
    # print('new/org=%d/%d=%f' %(newStreamSize, orgStreamSize, newStreamSize/orgStreamSize))
    return newStreamSize/orgStreamSize

if __name__ == '__main__':
    classname = 'walk'
    main_stat(classname)





