# from sklearn import datasets
from e2sr import Video_test

DATA_DIR = "/home/yuzhongkai/E2SR/datasets/"
dataset = "Vid4_loss"
videoname = "calendar"

video_test = Video_test(DATA_DIR, dataset, videoname)
# video_test = Video_test(DATA_DIR, dataset, videoname, 700, 8)
video_test.cloud_server(isPar=True)  # run mv search algorithm, using parallel computing
video_test.device()  # run reconstruction algorithm
psnr = video_test.test_PSNR()  # test psnr
bs_size = video_test.test_bs_size()  # test bit-stream size
print(psnr, bs_size)