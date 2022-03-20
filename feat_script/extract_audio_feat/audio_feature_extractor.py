import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = "1" # set gpu number
import numpy as np
import tensorflow as tf

import vggish_input
import vggish_params
import vggish_slim
import h5py
import contextlib
import wave


# get audio length
def get_audio_len(audio_file):
    # audio_file = os.path.join(audio_path, audio_name)
    with contextlib.closing(wave.open(audio_file, 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        wav_length = int(frames / float(rate))
        # print("wave_len: ", wav_length)

        return wav_length



# Paths to downloaded VGGish files.
checkpoint_path = 'vggish_model.ckpt'
pca_params_path = 'vggish_pca_params.npz'
# num_secs = 60 # length of the audio sequence. Videos in our dataset are all 10s long.
freq = 1000
sr = 44100


audio_dir = "./data/audio/" # .wav audio files
save_dir = "./data/feats/vggish/"


lis = sorted(os.listdir(audio_dir))
len_data = len(lis)
print(len_data)

i = 0

for n in range(len_data):
    i += 1

    # save file
    outfile = os.path.join(save_dir, lis[n][:-4] + '.npy')
    if os.path.exists(outfile):
        print("\nProcessing: ", i, " / ", len_data, " ----> ", lis[n][:-4] + '.npy', " is already exist! ")
        continue

    '''feature learning by VGG-net trained by audioset'''
    audio_index = os.path.join(audio_dir, lis[n]) # path of your audio files
    num_secs = get_audio_len(audio_index)
    print("\nProcessing: ", i, " / ", len_data, " --------> video: ", lis[n], " ---> sec: ", num_secs)

    input_batch = vggish_input.wavfile_to_examples(audio_index, num_secs)
    np.testing.assert_equal(
        input_batch.shape,
        [num_secs, vggish_params.NUM_FRAMES, vggish_params.NUM_BANDS])

    
    
    # Define VGGish, load the checkpoint, and run the batch through the model to
    # produce embeddings.
    # with tf.Graph().as_default(), tf.Session() as sess:
    with tf.Graph().as_default(), tf.compat.v1.Session() as sess:
        vggish_slim.define_vggish_slim()
        vggish_slim.load_vggish_slim_checkpoint(sess, checkpoint_path)

        features_tensor = sess.graph.get_tensor_by_name(vggish_params.INPUT_TENSOR_NAME)
        embedding_tensor = sess.graph.get_tensor_by_name(vggish_params.OUTPUT_TENSOR_NAME)
        [embedding_batch] = sess.run([embedding_tensor], feed_dict={features_tensor: input_batch})
        #print('VGGish embedding: ', embedding_batch[0])
        # outfile = os.path.join(save_dir, lis[n][:-4] + '.npy')
        np.save(outfile, embedding_batch)
        #audio_features[i, :, :] = embedding_batch
        print(" save info: ", lis[n][:-4] + '.npy', " ---> ", embedding_batch.shape)

        i += 1

print("\n---------------------------------- end ----------------------------------\n")


