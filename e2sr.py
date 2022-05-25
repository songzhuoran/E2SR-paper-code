#import torch
import cv2
import numpy as np
# import glob
# from PIL import Image
# import openpyxl
# import xlwt
import csv
import math
import os
from progressbar import *
import re
import matplotlib.pyplot as plt
from multiprocessing import Process, Semaphore
from multiprocessing import Pool
from multiprocessing import Manager
from PSNR import *
from stat_stream import *


class Video_test:
    def __init__(self, DATA_DIR, dataset, videoname, t1=1000, t2=100) -> None:
        self.dataset = dataset
        self.videoname = videoname
        self.t1 = t1
        self.t2 = t2
        self.sr_frame_h, self.sr_frame_w = 0, 0
        self.bflist, self.pflist = [], []
        self.SR_PICS, self.SRLL_PICS, self.SR_PICS_1d= {}, {}, {}
        self.GT_PICS, self.GT_PICS_1d = {}, {} 
        self.Res_data = []
        self.DATA_DIR = DATA_DIR
        self.BS_OUT_DIR = "./bs_out"
        self.FRAME_OUT_DIR = "./frame_out"
        self.frameidFlag, self.picsFlag = False, False  #  False: never load such data

    def get_frame_info(self):
        GT_PICS_DIR = "%s/%s/GT/" % (self.DATA_DIR, self.dataset)
        pic_names = os.listdir(GT_PICS_DIR + self.videoname)
        img = cv2.imread(GT_PICS_DIR + self.videoname + '/' + pic_names[0], -1)
        self.sr_frame_h, self.sr_frame_w, _ = img.shape  # get shape of pictures

    def fetch_bp_frame_id(self):
        '''
        fetch idx of b frames into bflist
        fetch idx of p frames into pflist
        '''

        if not self.frameidFlag:
            IDX_DIR = "%s/%s/Info_BIx4/idx/" % (self.DATA_DIR, self.dataset)
            with open(IDX_DIR+"b/" + self.videoname, "r") as file:
                for row in file:
                    self.bflist.append(int(row)-1)
            print('bflist: ', self.bflist)

            with open(IDX_DIR+"p/"+self.videoname, "r") as file:
                for row in file:
                    self.pflist.append(int(row)-1)
            print('plist: ', self.pflist)
            self.frameidFlag = True
  
    def fetch_pics(self):
        '''
        load all GT and SR pics to avoid repetitive I/O
        '''
        if not self.picsFlag:
            SR_PICS_DIR = "%s/%s/SR/" % (self.DATA_DIR, self.dataset)
            SRLL_PICS_DIR = "%s/%s/SR_lossless/" % (self.DATA_DIR, self.dataset)
            # GT_PICS_DIR = "%s/%s/GT/" % (self.DATA_DIR, self.dataset)
            for picId in self.bflist + self.pflist:
                self.SR_PICS[picId] = cv2.imread(SR_PICS_DIR + self.videoname + "/%08d.png" % (picId+1)) 
                self.SRLL_PICS[picId] = cv2.imread(SRLL_PICS_DIR + self.videoname + "/%08d.png" % (picId+1))
                # self.GT_PICS[picId] = cv2.imread(GT_PICS_DIR + self.videoname + "/%08d.png" % (picId+1))
                # self.GT_PICS_1d[picId] = cv2.imread(GT_PICS_DIR + self.videoname + "/%08d.png" % (picId+1), 0)
                self.SR_PICS_1d[picId] = cv2.imread(SR_PICS_DIR + self.videoname + "/%08d.png" % (picId+1), 0)	
        self.picsFlag = True

    def fetch_mv_res(self):
        '''
            load the whole res data file into a py list to prevent redundant loading
        '''
        self.Res_data = []
        with open("%s/%s/E2SR_cop_%s_%d_%d.csv" %(self.BS_OUT_DIR, self.dataset, self.videoname, self.t1, self.t2), "r") as file:
            reader = csv.reader(file)
            for item in reader:
                self.Res_data.append(item)

    def mv_search(self, isPar=True):
        def mycallback(lines):
            with open(fname, 'a+', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(lines)

        os.system("mkdir -p %s/%s" %(self.BS_OUT_DIR, self.dataset))
        if not isPar:
            self.mv_search_kernel()
        else:
            mv_loss_dir = "%s/%s/Info_BIx4/mvs/%s_loss.csv" %(self.DATA_DIR, self.dataset, self.videoname)
            mvs = np.loadtxt(open(mv_loss_dir, "rb"), delimiter=",", skiprows=0).astype(int)
            mvs_slice = {x: [] for x in self.bflist}
            for mv in mvs:
                curId = int(float(mv[0]))
                if curId in self.bflist:
                    mvs_slice[curId].append(mv)
            fname = "%s/%s/E2SR_%s_%d_%d.csv" %(self.BS_OUT_DIR, self.dataset, self.videoname, self.t1, self.t2)
            os.system("rm -f %s" % fname)
            mypool = Pool(32)  # 32 cores in our server
            for fidx in self.bflist:
                mypool.apply_async(func=self.mv_search_kernel_par, args=(fidx, mvs_slice[fidx]), callback=mycallback)
                print("generating %s video, frame: %d" %(self.videoname, fidx))
            mypool.close()
            mypool.join()

    def mv_search_kernel(self):
        mv_loss_dir = "%s/%s/Info_BIx4/mvs/%s_loss.csv" %(self.DATA_DIR, self.dataset, self.videoname)
        mvs = np.loadtxt(open(mv_loss_dir, "rb"), delimiter=",", skiprows=0).astype(int)
        a = 0
        widgets = ['Progress: ', Percentage(), ' ', Bar('#'), ' ', Timer(), ' ', ETA()]
        pbar = ProgressBar(widgets=widgets).start()
        newmvs = []
        for i, mv in enumerate(mvs):
            pbar.update(i/(len(mvs)-1)*100)
            a+=1
            curId = mv[0]
            # refId = mv[1]
            blockw, blockh, curx, cury, refx, refy = mv[2:8]
            ## use canny to find edges in pictures
            # img_canny = GT_PICS_1d[curId]
            img_canny = self.SR_PICS_1d[curId]

            canny_res = cv2.Canny(img_canny, self.t1, self.t1) # image after canny,for walk
            srcurx = 4 * curx
            srcury = 4 * cury
            srrefx = 4 * refx
            srrefy = 4 * refy
            srBlockw = 4 * blockw
            srBlockh = 4 * blockh

            ## 使用边界来决定哪些块需要切分
            self.prediction_unit_canny(mv,srcurx,srcury,srrefx,srrefy,srBlockw,srBlockh,canny_res,newmvs)
        pbar.finish()
        # os.system("mkdir -p %s/%s" %(self.BS_OUT_DIR, self.dataset))
        fname = "%s/%s/E2SR_%s_%d_%d.csv" %(self.BS_OUT_DIR, self.dataset, self.videoname, self.t1, self.t2)
        with open(fname, 'w', newline='') as mvsf:
            writer = csv.writer(mvsf)
            writer.writerows(newmvs)

    def mv_search_kernel_par(self, fidx, mvs):
        mvsf = []
        # print("frame %d, length: %d" %(fidx, len(mvs)))
        img_canny = self.SR_PICS_1d[fidx]
        canny_res = cv2.Canny(img_canny, self.t1, self.t1) # image after canny
        for mv in mvs:
            curId = mv[0]
            if int(float(curId)) != fidx:
                continue
            blockw, blockh, curx, cury, refx, refy = mv[2:8]
            srcurx = 4 * curx
            srcury = 4 * cury
            srrefx = 4 * refx
            srrefy = 4 * refy
            srBlockw = 4 * blockw
            srBlockh = 4 * blockh
            ## 使用边界来决定哪些块需要切分
            self.prediction_unit_canny(mv,srcurx,srcury,srrefx,srrefy,srBlockw,srBlockh,canny_res,mvsf)
        print("frame %d finished" % fidx)
        return mvsf 

    def prediction_unit_canny(self, mv,srcurx,srcury,srrefx,srrefy,srBlockw,srBlockh,canny_res,f):

        def slideIndex(block):
            return np.sum(block)/255

        def prediction_unit_kernel(mv,srcurx,srcury,srrefx,srrefy,srBlockw,srBlockh,canny_res):
            curId = mv[0]
            tmp_block = canny_res[srcury:srcury + srBlockh, srcurx:srcurx + srBlockw]
            if slideIndex(tmp_block) >= self.t2 and srBlockw > 4 and srBlockh > 4:
                srBlockw = int(srBlockw / 2)
                srBlockh = int(srBlockh / 2)
                prediction_unit_kernel(mv,srcurx+srBlockw,srcury,srrefx+srBlockw,srrefy,srBlockw,srBlockh,canny_res)
                prediction_unit_kernel(mv,srcurx,srcury+srBlockh,srrefx,srrefy+srBlockh,srBlockw,srBlockh,canny_res)
                prediction_unit_kernel(mv,srcurx+srBlockw,srcury+srBlockh,srrefx+srBlockw,srrefy+srBlockh,srBlockw,srBlockh,canny_res)
                prediction_unit_kernel(mv,srcurx,srcury,srrefx,srrefy,srBlockw,srBlockh,canny_res)
            else:
                refs = sorted(self.pflist, key=lambda x : abs(x-curId))
                refs = refs[0:2]
    
                curFrame = self.SRLL_PICS[curId].astype(int)  

                srFrameh, srFramew, _ = curFrame.shape

                curBlock = np.zeros((srBlockh, srBlockw, 3))
                partCurBlock = curFrame[srcury: srcury+srBlockh, srcurx:srcurx+srBlockw]
                act_h, act_w, _ = partCurBlock.shape
                curBlock[0:act_h, 0:act_w] = curFrame[srcury:srcury+act_h, srcurx:srcurx+act_w]

                bestRefId, bestRefFrame, bestRefBlock, bestRefxy = None, None, None, None
                bestDiff = float('Inf')

                sRange = 1
                s = 1
                if srrefx<0:
                    sRange = max(sRange,abs(srrefx)+act_h+1)
                if srrefy<0:
                    sRange = max(sRange,abs(srrefy)+act_h+1)
                if srrefx>(srFramew-srBlockw):
                    sRange = max(sRange,srrefx-srFramew+act_h+1)
                if srrefy>(srFrameh-srBlockh):
                    sRange = max(sRange,srrefy-srFrameh+act_h+1)

                ymin = srrefy - sRange
                ymax = srrefy + sRange
                xmin = srrefx - sRange
                xmax = srrefx + sRange

                for ref in refs:
                    refFrame = self.SR_PICS[ref]
                    for fy in range(ymin, ymax, s):
                        for fx in range(xmin, xmax, s):
                            refBlock = np.zeros((srBlockh, srBlockw, 3))
                            partRefBlock = refFrame[max(fy, 0): max(fy+srBlockh, 0),
                                            max(fx, 0): max(fx+srBlockw, 0)]
                            act_h, act_w, _ = partRefBlock.shape

                            frameXmin = max(0, fx)
                            frameXmax = max(0, min(fx+srBlockw, srFramew))
                            blockXmin = frameXmin - fx
                            blockXmax = blockXmin + act_w

                            frameYmin = max(0, fy)
                            frameYmax = max(0, min(fy+srBlockh, srFrameh))
                            blockYmin = frameYmin - fy
                            blockYmax = blockYmin + act_h

                            refBlock[blockYmin: blockYmax, blockXmin: blockXmax] = \
                                refFrame[frameYmin: frameYmax, frameXmin:frameXmax]

                            diff = np.sum((curBlock - refBlock) ** 2)
                            if diff < bestDiff:
                                bestRefId = ref
                                bestRefBlock = refBlock
                                bestRefxy = (fx, fy)
                                bestDiff = diff

                line = [curId, bestRefId, srBlockw, srBlockh, srcurx, srcury, bestRefxy[0], bestRefxy[1]]
                res_3d = curBlock - bestRefBlock
                res_1d = res_3d.reshape(-1)
                line.extend(res_1d)
                f.append(line)


        prediction_unit_kernel(mv,srcurx,srcury,srrefx,srrefy,srBlockw,srBlockh,canny_res)

    def compress_mv(self):
        '''
            compress residual and create a new csv file
        '''
        
        with open("%s/%s/E2SR_%s_%d_%d.csv" %(self.BS_OUT_DIR, self.dataset, self.videoname, self.t1, self.t2), "r") as f1:
            reader = csv.reader(f1)
            org_file = []
            for data in reader:
                org_file.append(data)
                
        f2 = open("%s/%s/E2SR_cop_%s_%d_%d.csv" %(self.BS_OUT_DIR, self.dataset, self.videoname, self.t1, self.t2), 'w', newline='')
        writer = csv.writer(f2)
        widgets = ['Progress: ', Percentage(), ' ', Bar('#'), ' ', Timer(), ' ', ETA()]
        pbar = ProgressBar(widgets=widgets).start()
        for i, org_line in enumerate(org_file):
            pbar.update(i / (len(org_file) - 1) * 100)
            block_w, block_h, curx, cury, refx, refy = np.array(org_line[2:8]).astype(int)
            org_res_1d = np.array(org_line[8:]).astype(float).astype(np.int16)

            org_res_3d = org_res_1d.reshape((block_h, block_w, 3))
            comp_res_3d = cv2.resize(org_res_3d, dsize=None, fx=0.25, fy = 0.25, interpolation=cv2.INTER_AREA)
            comp_res_1d = comp_res_3d.reshape(-1)
            newline = org_line[0:8]
            newline.extend(comp_res_1d)
            writer.writerow(newline)
        pbar.finish()
        f2.close()

    def bframe_gen_kernel(self, fcnt):
        '''
            使用下采样后的res并进行上采样
        '''
        print(fcnt, len(self.Res_data))
        new_Res_data = self.Res_data.copy()
        bframe_img_sr = np.zeros((self.sr_frame_h, self.sr_frame_w, 3))
        for row in self.Res_data:
            if int(float(row[0])) == fcnt:
                new_Res_data.remove(row)#每次减少Res_data数目，加速整体过程
                refFrame = self.SR_PICS[int(float(row[1]))]
                frameh, framew, _ = refFrame.shape
                blockw, blockh, curx, cury, refx, refy = np.array(row[2:8]).astype(float).astype(int)
                res_1d = np.array(row[8:]).astype(float).astype(np.int16)
                # print(blockw, blockh)
                if blockh >=4 and blockw >=4:

                    res_3d = res_1d.reshape((int(blockh/4), int(blockw/4), 3))
                    res_3d_ext = cv2.resize(res_3d, dsize=None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
                # elif blockh ==2 and blockw ==2:
                # 	res_3d = res_1d.reshape((int(blockh/2), int(blockw/2), 3))
                # 	res_3d_ext = cv2.resize(res_3d, dsize=None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                # else:
                # 	res_3d_ext = res_1d
                    

                refBlock = np.zeros((blockh, blockw, 3))
                refActh, refActw, _ = refFrame[max(refy, 0): max(refy + blockh, 0),
                            max(refx, 0): max(refx + blockw, 0)].shape
                frameXmin = max(0, refx)
                frameXmax = max(0, min(refx + blockw, framew))  

                blockXmin = frameXmin - refx
                blockXmax = blockXmin + refActw

                frameYmin = max(0, refy)
                frameYmax = max(0, min(refy + blockh, frameh))
                blockYmin = frameYmin - refy
                blockYmax = blockYmin + refActh

                refBlock[blockYmin: blockYmax, blockXmin: blockXmax] = \
                    refFrame[frameYmin: frameYmax, frameXmin:frameXmax]

                curBlock = refBlock + res_3d_ext
                # curBlock = refBlock
                curActh, curActw, _ = refFrame[max(cury, 0): max(cury + blockh, 0),
                            max(curx, 0): max(curx + blockw, 0)].shape
                bframe_img_sr[cury:cury+curActh, curx:curx+curActw] = curBlock[0:curActh, 0:curActw]

        cv2.imwrite("%s/%s/%s_%d_%d/%08d.png" %(self.FRAME_OUT_DIR, self.dataset, self.videoname, self.t1, self.t2, (fcnt+1)), bframe_img_sr)
        self.Res_data = new_Res_data

    def bframe_gen(self):
        # os.system("mkdir -p %s/%s/%s_%d_%d" %(self.FRAME_OUT_DIR, self.dataset, self.videoname, self.t1, self.t2))
        for fcnt in self.bflist:
            # print(i)
            self.bframe_gen_kernel(fcnt)

    def cloud_server(self, isPar=True):
        self.fetch_bp_frame_id()
        self.fetch_pics()
        self.mv_search(isPar)
        self.compress_mv()

    def device(self):
        self.fetch_bp_frame_id()
        self.fetch_pics()
        self.get_frame_info()
        self.fetch_mv_res()
        new_dir = "%s/%s/%s_%d_%d" %(self.FRAME_OUT_DIR, self.dataset, self.videoname, self.t1, self.t2)
        os.system("mkdir -p %s" %(new_dir))
        sr_dir = "%s/%s/SR/%s" % (self.DATA_DIR, self.dataset, self.videoname)
        os.system("cp -r %s/* %s/" %(sr_dir, new_dir))
        # os.system("cp -r %s/* %s" %(sr_dir, new_dir))
        self.bframe_gen()

    def test_PSNR(self):
        path1 = "%s/%s/GT/%s" % (self.DATA_DIR, self.dataset, self.videoname)
        path2 = "%s/%s/%s_%d_%d" % (self.FRAME_OUT_DIR, self.dataset, self.videoname, self.t1, self.t2)
        score = quick_PSNR(path1, path2)  # test PSNR
        return score

    def test_bs_size(self):
        self.fetch_bp_frame_id()
        org_bs = "%s/%s/Info_BIx4/mvs/%s_loss.csv" %(self.DATA_DIR, self.dataset, self.videoname)
        new_bs = "%s/%s/E2SR_cop_%s_%d_%d.csv" %(self.BS_OUT_DIR, self.dataset, self.videoname, self.t1, self.t2)
        bs_size = quick_stat_class(self.bflist, org_bs, new_bs)
        return bs_size