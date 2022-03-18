# -*- coding: utf-8 -*-

'''
Function: Extract video frames.
'''

from video2frames import video_processing

if __name__ == "__main__":

    dir_path = '../../data/AVQA/videos/'
    dst_path = '../../data/AVQA/frames/'

    process_count = 0
    fix_second = 0
    fps_count = 1

    video_processing(dir_path, dst_path, process_count, fix_second, fps_count)
