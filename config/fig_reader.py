import tifffile
import numpy as np

# 打开 TIFF 文件并逐页查看内容
with tifffile.TiffFile('/data/staff/tomograms/users/zhehu/pixelnerf/dataset/new_data/AlGe10_040_240k_corrected.tif') as tif:
    # 使用列表推导式读取每个页面的数据
    data = np.array([page.asarray() for page in tif.pages], dtype=np.float32)
data = data.reshape((30,200,280,528))
np.save('/data/staff/tomograms/users/zhehu/pixelnerf/dataset/AlGe.npy',data)