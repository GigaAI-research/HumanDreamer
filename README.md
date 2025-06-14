<div align="center">   
  
# HumanDreamer: Generating Controllable Human-Motion Videos via Decoupled Generation  
**CVPR 2025**

</div>

<!-- [Project Page](https://humandreamer.github.io/)  | [Paper](https://arxiv.org/abs/2503.24026)  -->
[![Website](https://img.shields.io/badge/Project-Website-0073e6)](https://humandreamer.github.io/)
[![Paper](https://img.shields.io/badge/arXiv-PDF-b31b1b)](https://arxiv.org/abs/2503.24026)
<!-- [![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT) -->


<div align="center">
<img src="./assets/main_demo.png" alt="Teaser Image" width="60%">
</div>

<!-- ## Method Overview

Training pipeline of the proposed *Text-to-Pose* generation. Pose data are encoded in latent space via the Pose VAE, which are then processed by the proposed MotionDiT, where local feature aggregation and global attention are utilized to capture information from the entire pose sequence. Finally, the LAMA loss is calculated via the proposed CLoP, which enhances the training of MotionDiT.

![motiondit](./assets/motiondit.png) -->

<!-- The pipeline of *Pose-to-Video*.

![pose2video](./assets/pose2video.png) -->
# Project Status
- [2025/4/5]： ✅Repository Initialization.
- [2025/6/14]： ✅Release MotionVid dataset.
- 🛠️TODO: Release code.
---

# MotionVid Dataset
MotionVid is a dataset containing 1.2M text-pose-video pairs. The videos are sourced from the internet and public datasets, the poses are extracted using [DWPose](https://github.com/IDEA-Research/DWPose), and the text descriptions are generated using [ShareGPT4Video](https://github.com/ShareGPT4Omni/ShareGPT4Video).

**Note**: The repository does **not include video files** due to licensing restrictions. If you require the video files, you must download them separately from their respective public sources. Additionally, a small portion of the data (≈9%) cannot be made public due to policies considerations.

We have divided the data from different sources into multiple subsets. Each subset stores basic data information in `.pkl` format and pose information in `.mdb` format. 

You can download **MotionVid** dataset [**here**](https://huggingface.co/datasets/chuanshuogushi/MotionVid).


Dataset includes the following fields:

- **`data_index`**: Index of the sample.
- **`prompt`**: Description of the activity performed by the person in the video.
- **`video_height`**: Original height of the video.
- **`video_width`**: Original width of the video.
- **`video_length`**: Length of the video sequence.
- **`video_path`**: Name of the video file in the public dataset, which can be used to locate and download the video from its source.
- **`poses`**: Human keypoint information.
- **`poses_scores`**: Confidence scores for the keypoints.

<!-- ## Usage of Dataset
You can use the `load_dataset` method to load and view the data.
```
dataset = load_dataset('path/to/motionvid/subset0')
print(dataset[0].keys())
``` -->

<!-- # Code Release
## MotionDiT
Under reconstrucion.
### Train
### Inference
## PoseVAE
## CLoP -->
---

<!-- ## Installation
****
### Requirements

- Python 3.9+
- PyTorch 2.x
- CUDA-capable GPU
- Other dependencies listed in `requirements.txt` -->

<!-- ### Setup

```bash
# Clone the repo
git clone https://github.com/yourname/humandreamer.git 
cd humandreamer

# Install dependencies
pip install -r requirements.txt

# (Optional) Setup virtual environment
python -m venv venv && source venv/bin/activate


--- -->

## License
All the data and code within this repo are under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).

## BibTeX

```bibtex
@article{wang2025humandreamer,
  title={HumanDreamer: Generating Controllable Human-Motion Videos via Decoupled Generation}, 
  author={Boyuan Wang and Xiaofeng Wang and Chaojun Ni and Guosheng Zhao and Zhiqin Yang and Zheng Zhu and Muyang Zhang and Yukun Zhou and Xinze Chen and Guan Huang and Lihong Liu and Xingang Wang},
  journal={arXiv preprint arXiv:2503.24026},
  year={2025}
}