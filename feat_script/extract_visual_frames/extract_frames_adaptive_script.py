# -*- coding: utf-8 -*-

'''
Function: Extract video frames.

dir_path: Video storage path.
dst_path: Storage path of the extracted frame.
process_count: The number of threads, where 0 means the maximum number of threads.
fps_count: Number of frames extracted per second.
fix_second: Fix the video length, where 0 means the actual length.
'''

from extract_visual_frames/extract_frames_adaptive import video_processing

if __name__ == "__main__":

    dir_path = './data/videos/'
    dst_path = './data/frames/'

    process_count = 0
    fix_second = 60
    fps_count = 1

    video_processing(dir_path, dst_path, process_count, fix_second, fps_count)
