# -*- coding: utf-8 -*-
'''
Author: Guangyao Li
E-mail: guangyaoli@ruc.edu.cn
Date: 17/4/2021

Functions:

输入参数： 
- 视频原始路径 (dir_paht = "")
- 提帧目标路径 (dst_path = "")
- 每秒取几帧（frame = 1 or frame = 4)
- 固定or不固定（固定的话写参数N，不固定就是四舍五入取整）
- 线程数（默认最大，否则写参数J）


'''


import os
import logging
import time
import sys
import multiprocessing
from imageio import imsave
from moviepy.editor import VideoFileClip, concatenate_videoclips
import warnings
import cv2

warnings.filterwarnings('ignore')

# save log
def log_config(logger=None, file_name="log"):
    if logger is None:
        logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    if not os.path.exists("log"):
        os.makedirs("log")
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    if file_name is not None:
        fh = logging.FileHandler("log/%s-%s.log" % (file_name, time.strftime("%Y-%m-%d")), mode='a')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger


def is_generate(out_path, dst=10):
    if not os.path.exists(out_path):
        return False
    folder_list = os.listdir(out_path)
    jpg_number = 0
    for file_name in folder_list:
        if file_name.strip().lower().endswith('.jpg'):
            jpg_number += 1
    return jpg_number >= dst


def fixed_video(clip, video_len_pre):
    if clip.duration >= video_len_pre:
        return clip
    t_start = int(clip.duration)
    if t_start == clip.duration:
        t_start = -1
    last_clip = clip.subclip(t_start)
    final_clip = clip
    while final_clip.duration < video_len_pre:
        final_clip = concatenate_videoclips([final_clip, last_clip])
    return final_clip


def read_frame(reader, pos):
    if not reader.proc:
        reader.initialize()
        reader.pos = pos
        reader.lastread = reader.read_frame()

    if pos == reader.pos:
        return reader.lastread
    elif (pos < reader.pos) or (pos > reader.pos + 100):
        reader.initialize()
        reader.pos = pos
    else:
        reader.skip_frames(pos - reader.pos - 1)
    result = reader.read_frame()
    reader.pos = pos
    return result

def compute_numbers(outpath): 
    folder_list = os.listdir(outpath)
    folder_list.sort()

    jpg_number = 0
    pkl_number = 0
    for file_name in folder_list:
        if os.path.splitext(file_name)[-1].lower() == ".jpg":
            jpg_number += 1
        if os.path.splitext(file_name)[-1].lower() == ".pkl":
            pkl_number += 1

    if jpg_number != 10 or pkl_number != 10:
        return 1
    else:
        return 0

def deal_video(video_file, out_path, fix_second, fps_count):

    # total_temp = compute_numbers(out_path)
    total_temp = 1
    if total_temp == 0:
        return
    else:
        try:
            if is_generate(out_path):
                return
            # if os.path.exists(out_path):
            #     print(out_path, " is already extract! ****")
            #     return
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            if not os.path.isfile(video_file):
                logger.error("deal video error, %s is not a file", video_file)
                return
            with VideoFileClip(video_file) as clip:
                step = 1

                reader = clip.reader
                fps = clip.reader.fps
                total_frames = reader.nframes

                last_frames = int(total_frames % fps)

                if last_frames == 0:
                    last_frames = int(fps)
                last_start = total_frames - last_frames


                save_frame_index_arr = []

                video_len_pre = fix_second
                if fix_second == 0:
                    video_len_pre = round(total_frames/fps)

                video_len_pre = video_len_pre * fps_count

                for i in range(video_len_pre):
                    absolute_frame_pos = round((1 / (2*fps_count) + i / fps_count) * fps)

                    if absolute_frame_pos > total_frames:
                        relative_frame_pos = last_start + 1 + ((absolute_frame_pos - last_start - 1) % last_frames)
                    else:
                        relative_frame_pos = absolute_frame_pos

                    save_frame_index_arr.append(relative_frame_pos)

                save_frame_map = {}
                loop_arr = list(set(save_frame_index_arr))
                loop_arr.sort()
                for i in loop_arr:
                    if i not in save_frame_map:
                        im = read_frame(reader, i)
                        save_frame_map[i] = im

                success_frame_count = 0
                for i in range(len(save_frame_index_arr)):
                    try:
                        out_file_name = os.path.join(out_path, "{:08d}.jpg".format(i + 1))
                        im = save_frame_map[save_frame_index_arr[i]]
                        imsave(out_file_name, im)
                        success_frame_count += 1
                    except Exception as e:
                        logger.error("(%s) save frame(%s) error", video_file, str(i + 1), e)
                log_str = "video(%s) save frame, save count(%s) total(%s) fps(%s) %s, "
                if success_frame_count == video_len_pre:
                    logger.debug(log_str, video_file, success_frame_count, total_frames, fps, save_frame_index_arr)
                else:
                    logger.error(log_str, video_file, success_frame_count, total_frames, fps, save_frame_index_arr)

        except Exception as e:
            logger.error("deal video(%s) error", video_file, e)


def process_dir_path_class(param):
    dir_path_class = param["dir_path_class"]
    dst_path_class = param["dst_path_class"]
    fix_second = param["fix_second"]
    fps_count = param["fps_count"]
    try:
        print("\n-----------------------------------------\n")
        files = os.listdir(dir_path_class)
        files.sort()
        cnt = 0
        total_cnt = len(files)
        for video_file in files:
            name, ext = os.path.splitext(video_file)
            out_path = os.path.join(dst_path_class, name)
            cnt += 1
            # deal_video(os.path.join(dir_path_class, video_file), os.path.join(dst_path_class, name), fix_second, fps_count)
            if os.path.exists(out_path):
                print("Progress: ", cnt, " / ", total_cnt, " ------- id: ", video_file, " is already extract!")
                continue
            else:
                deal_video(os.path.join(dir_path_class, video_file), out_path, fix_second, fps_count)
            print("Progress: ", cnt, " / ", total_cnt, " ------- id: ", video_file)

    except Exception as e:
        logger.error("process(%s) error", dir_path_class, e)


def deal_dir(dir_path, dst_path, fix_second, fps_count, pool=None):
    print("----------------------------------")

    logger.info("----- deal dir: (%s), to path: (%s) -----", dir_path, dst_path)
    request_param = []

    param = {'dir_path_class': dir_path, 'dst_path_class': dst_path, "fix_second": fix_second, "fps_count": fps_count}
    if pool is None:
        process_dir_path_class(param)
    else:
        request_param.append(param)

    if pool is not None:
        pool.map(process_dir_path_class, request_param)
        pool.close()
        pool.join()


logger = log_config()

def video_processing(dir_path, dst_path, process_count, fix_second, fps_count):

    cpu_count = multiprocessing.cpu_count()
    if process_count == 0:
        # cpu_count = multiprocessing.cpu_count()
        process_count = cpu_count * 2 - 1

    logger.info("cpu count is {}, create {}, process pool".format(cpu_count, process_count))
    pool = multiprocessing.Pool(process_count)

    deal_dir(dir_path, dst_path, fix_second, fps_count, pool)
    print("--------------------------- end!-------------------------\n")