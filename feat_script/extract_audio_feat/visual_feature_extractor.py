import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import h5py
import sys
import cv2
import pylab
import imageio
from keras.layers import GlobalAveragePooling2D
from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model


def video_frame_sample(frame_interval, video_length, sample_num):
    num = []
    for l in range(video_length):

        for i in range(sample_num):
            num.append(int(l * frame_interval + (i * 1.0 / sample_num) * frame_interval))

    return num


base_model = VGG19(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('block5_pool').output) # vgg pool5 features

# path of your dataset
video_dir = "/home/cxu-serve/u1/ytian21/dat/AVE10K/videos"
lis = sorted(os.listdir(video_dir))

len_data = len(lis)
video_features = np.float32(np.zeros([len_data, 10, 7, 7, 512])) # 10s long video
t = 10 # length of video
sample_num = 8 # frame number for each second
print(len_data)
c = 0
for num in range(len_data):

    '''feature learning by VGG-net'''
    video_index = os.path.join(video_dir, lis[num]) # path of videos
    vid = imageio.get_reader(video_index, 'ffmpeg')

    imgs = []
    try:
        for i, im in enumerate(vid):
            x_im = cv2.resize(im, (224, 224))
            imgs.append(x_im)
    except RuntimeError:
        pass
    vid.close()

    vid_len = len(imgs)
    frame_interval = int(vid_len / t)

    frame_num = video_frame_sample(frame_interval, t, sample_num)

    print(video_index, vid_len, frame_num)

    extract_frame = []
    for n in frame_num:
        extract_frame.append(imgs[n])

    feature = np.zeros(([10, 8, 7, 7, 512]))
    for j in range(len(extract_frame)):
        y_im = extract_frame[j]

        x = image.img_to_array(y_im)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        pool_features = np.float32(model.predict(x))

        tt = int(j / sample_num)
        video_id = j - tt * sample_num
        feature[tt, video_id, :, :, :] = pool_features
    feature_vector = np.float32(np.mean(feature, axis=(1))) # averaging features for 16 frames in each second
    video_features[num, :, :, :, :] = feature_vector
    c += 1

# save the visual features into one .h5 file. If you have a very large dataset, you may save each feature into one .npy file
with h5py.File('/home/cxu-serve/u1/ytian21/dat/AVE10K/feature/AVEII_vgg19_pool5.h5', 'w') as hf:
    hf.create_dataset("dataset", data=video_features)