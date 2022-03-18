import os
import ffmpeg
import subprocess
from pydub import AudioSegment

'''
Function: Convert .mp3 format to .wav
'''

def MP32WAV(mp3_file, mp3_file_path, dst_path):

	save_path_name = os.path.join(dst_path, mp3_file.replace(".mp3", ".wav"))
	mp3wavCmd = "ffmpeg -i" + " " + mp3_file_path + " " + save_path_name + " " + " -loglevel quiet"
	subprocess.call(mp3wavCmd, shell=True)


def MP32WAV_pydub(mp3_file, mp3_file_path, dst_path):

	save_path_name = os.path.join(dst_path, mp3_file.replace(".mp3", ".wav"))
	MP3_File = AudioSegment.from_mp3(file=mp3_file_path)
	MP3_File.export(save_path_name, format="wav")


if __name__ == "__main__":

	dir_path = "../../data/AVQA/audio_mp3/"
	dst_path = "../../data/AVQA/audio_wav/"

	dir_path_list = os.listdir(dir_path)
	dir_path_list.sort()

	dir_len = len(dir_path_list)
	cnt = 0

	for mp3_file in dir_path_list:
		mp3_file_path = os.path.join(dir_path, mp3_file)
		MP32WAV(mp3_file, mp3_file_path, dst_path)
		# MP32WAV_pydub(mp3_file, mp3_file_path, dst_path)
		cnt = cnt + 1
		print("--->>> Current progress: ", cnt, " / ", dir_len)

	print("\n---------------------------- Finshed! ----------------------------")
