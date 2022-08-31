# -*- coding: utf-8 -*-

'''
Function: Extract video frames.
'''

from extract_visual_frames/extract_frames_adaptive import video_processing

if __name__ == "__main__":

    dir_path = './data/videos/'
    dst_path = './data/frames/'

    process_count = 0
    fix_second = 0
    fps_count = 1

    video_processing(dir_path, dst_path, process_count, fix_second, fps_count)
