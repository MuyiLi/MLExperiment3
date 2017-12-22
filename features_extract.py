from PIL import Image
import pylab
import numpy as np
import os
from feature import NPDFeature
import pickle

# 用于提取特征，若源文件不存在或已经存在特征文件会报错

def save(model, filename):
    with open(filename, "wb") as f:
        pickle.dump(model, f)

# 提取正例特征
pos_img_path = 'datasets\\original\\face\\'
features_face = []
pos_list_dir = os.listdir(pos_img_path)
count = 1
for filename in pos_list_dir:
	im = Image.open(pos_img_path+filename).convert('L').resize((24,24)) 
	im_array = np.array(im) 
	im_feature = NPDFeature(im_array).extract()
	print('pos:',count)
	count += 1
	features_face.append(im_feature)

save(features_face,'features_face')  


# 提取负例特征
neg_img_path = 'datasets\\original\\nonface\\'
features_nonface = []
neg_list_dir = os.listdir(neg_img_path)
count = 1
for filename in neg_list_dir:
	im = Image.open(neg_img_path+filename).convert('L').resize((24,24)) 
	im_array = np.array(im) 
	im_feature = NPDFeature(im_array).extract()
	print('neg:',count)
	count += 1
	features_nonface.append(im_feature)

save(features_nonface,'features_nonface')  
