

# Audio-Visual Question Answering (AVQA)

PyTorch code accompanies our CVPR 2022 paper:

**Learning to Answer Questions in Dynamic Audio-Visual Scenarios**

[Guangyao Li](https://ayameyao.github.io/), [Yake Wei](https://echo0409.github.io/), [Yapeng Tian](https://yapengtian.org/), [Chenliang Xu](https://www.cs.rochester.edu/~cxu22/), [Ji-Rong Wen](http://ai.ruc.edu.cn/szdw/68136712b556477db57c8ae66752768f.htm) and [Di Hu](https://dtaoo.github.io/index.html)

**Resources:**  [[Paper]](https://gewu-lab.github.io/MUSIC-AVQA/static/files/MUSIC-AVQA.pdf), [[Supplementary]](https://gewu-lab.github.io/MUSIC-AVQA/static/files/MUSIC-AVQA-supplementary.pdf),  [[Poster]](https://gewu-lab.github.io/MUSIC-AVQA/static/files/MUSIC-AVQA-poster.pdf), [[Video]](https://www.bilibili.com/video/BV1Br4y1q7YN/) 

**Project Homepage:**  [https://https://gewu-lab.github.io/MUSIC-AVQA/](https://gewu-lab.github.io/MUSIC-AVQA/)

---



## What's Audio-Visual Question Answering Task?

We focus on **audio-visual question answering (AVQA) task, which aims to answer questions regarding different visual objects, sounds, and their associations in videos**. The problem requires comprehensive multimodal understanding and spatio-temporal reasoning over audio-visual scenes.

<div  align="center">    
<img src="figs/avqa_white.png" width = "60%" />
</div>


## MUSIC-AVQA Dataset

The large-scale MUSIC-AVQA dataset of musical performance, which contains **45,867 question-answer pairs**, distributed in **9,290 videos** for over 150 hours. All QA pairs types are divided into **3 modal scenarios**, which contain **9 question types** and **33 question templates**. Finally, as an open-ended problem of our AVQA tasks, all 42 kinds of answers constitute a set for selection. 

- **QA examples**

  <div  align="center">    
  <img src="figs/st_avqa_pairs.png" width = "100%" />
  </div>



##  Model Overview

To solve the AVQA problem, we propose a spatio-temporal grounding model to achieve scene understanding and reasoning over audio and visual modalities. An overview of the proposed framework is illustrated in below  figure.

<div  align="center">    
<img src="figs/framework.png" width = "90%" />
</div>



## Requirements

```python
python3.6 +
pytorch1.6.0
tensorboardX
numpy
```



## Usage

1. **Clone this repo**

   ```python
   https://github.com/ayameyao/MUSIC-AVQA.git
   ```

2. **Download data**

   **Annotations (QA pairs, etc.)**

   - Available for download at [Google Drive (20MB zipped)](http://www.google.com/), [Baidu Drive (20MB zipped)](https://www.baidu.com/)
   - The annotation files are stored in JSON format. Each annotation file contains seven different keyword. And more detail see in [Project Homepage](https://ayameyao.github.io/MUSIC-AVQA/)

   **Features**

   - We use [VGGish](https://github.com/tensorflow/models/tree/master/research/audioset/vggish), [ResNet18](https://pytorch.org/docs/stable/torchvision/models.html), and [ResNet (2+1)D](https://pytorch.org/docs/stable/torchvision/models.html) to extract audio, 2D frame-level, and 3D snippet-level features, respectively. 

   - The audio and visual features of videos in the MUSIC-AVQA dataset can be download from this [Google Drive]() or [Baidu Drive](). 

   - The features are in the "./feats" folder.
   - 14x14 features, too large to share ... but we can extract from raw videos.

   **Download videos**

   - Raw videos (trimmed): Available at [Google Drive (50GB zipped)](http://www.google.com/) or [Baidu Drive (50GB zipped)](https://www.baidu.com/).

   - Download raw videos in the MUSIC-AVQA dataset. The downloaded videos will be in the data/MUSIC-AVQA_dataset/video folder. 

   - Pandas and ffmpeg libraries are required.


3. **Data pre-processing**

   Extract audio waveforms from videos. The extracted audios will be in the data/MUSIC-AVQA_dataset/audio folder. `moviepy library` is used to read videos and extract audios.

   ```python
   python feat_script/extract_audio_cues/extract_audio.py	
   ```

   Extract video frames from videos. The extracted frames will be in the data/MUSIC-AVQA_dataset/frames folder.

   ```python
   python feat_script/extract_visual_frames/extract_frames_adaptive_script.py
   ```

4. **Feature extraction** 

   Audio feature. `TensorFlow1.4` and `VGGish pretrained on AudioSet` is required. Feature file also can be found from [here](). 

   ```python
   python feat_script/extract_audio_feat/audio_feature_extractor.py
   ```

   2D visual feature. Pretrained models library is required.

   ```python
   python feat_script/eatract_visual_feat/extract_rgb_feat.py
   ```

   3D visual feature.

   ```python
   python feat_script/eatract_visual_feat/extract_3d_feat.py
   ```

   14x14 visual feature.

   ```python
   python feat_script/extract_visual_feat_14x14/extract_14x14_feat.py
   ```

5. **Baseline Model**

   Training

   ```python
   python net_baseline/main_qa_grd_baseline.py --mode train
   ```

   Testing

   ```python
   python net_baseline/main_qa_grd_baseline.py --mode test
   ```

6. **Our Audio-Visual Spatial-Temporal Model**

   Audio-Visual grounding generation

   ```python
   python grounding_gen/main_grd_gen.py
   ```

   Training

   ```python
   python net_ours_avst/main_ours_avst.py --mode train
   ```

   Testing

   ```python
   python net_ours_avst/main_ours_avst.py --mode test
   ```



## Results

1. **Audio-visual video question answering results** of different methods on the test set of MUSIC-AVQA. The top-2 results are highlighted. Please see the citations in the [[Paper]](https://ayameyao.github.io/MUSIC-AVQA/static/files/MUSIC-AVQA.pdf) for comparison methods.

   <div  align="center">    
   <img src="figs/exp1.png" width = "100%" />
   </div>
   
   
2. **Visualized spatio-temporal grounding results**

   We provide several visualized spatial grounding results. The heatmap indicates the location of sounding source. Through the spatial grounding results, the sounding objects are visually captured, which can facilitate the spatial reasoning.

   ```python
   python grounding_gen/main_grd_gen_vis.py
   ```
   
   <div  align="center">    
   <img src="figs/visualize.png" width = "100%" />
   </div>




## Citation

If you find this work useful, please consider citing it.

<pre><code>
@ARTICLE{Li2022Learning,
  title	= {Learning to Answer Questions in Dynamic Audio-Visual Scenarios},
  author	= {Guangyao li, Yake Wei, Yapeng Tian, Chenliang Xu, Ji-Rong Wen, Di Hu},
  journal	= {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year	= {2022},
  Url	= {https://doi.org/10.1007/s11263-021-01531-2}
}
</code></pre>



## Acknowledgement

This research was supported by Public Computing Cloud, Renmin University of China.




## License

This project is released under the [GNU General Public License v3.0](https://github.com/Mukosame/Zooming-Slow-Mo-CVPR-2020/blob/master/LICENSE).



