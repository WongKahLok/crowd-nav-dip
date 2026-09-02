# CrowdNav

## Setup
1. Install necessary library for the system
```
sudo apt update && sudo apt install cmake build-essential ffmpeg
```

2. Create Anconda environment
```terminal
conda env create -n crowdnav python=3.10
```

3. Install crowd_sim and crowd_nav into pip
```
pip install -r requirements.txt
```

4. Install [Python-RVO2](https://github.com/sybrenstuvel/Python-RVO2) library 
```
unzip Python-RVO2-main.zip
cd Python-RVO2-main
python setup.py build && python setup.py install
cd ..
pip install -e .
```

## Getting Started
1. Train a policy.
```
python train.py --policy sarl --gpu
```
2. Visualize a test case.
```
python test.py \
               --policy sarl \
               --model_dir data/output-trained \
               --phase test \
               --visualize \
               --test_case 0 \
               --video_file data/output-trained/test_case_0.mp4 \
               --static_map ../static_maps/land_shapes_sf_crop.png
```
```
python test.py \
               --policy sarl \
               --model_dir data/output-trained \
               --phase test \
               --visualize \
               --test_case 0 \
               --video_file data/output-trained/test_case_1.mp4 \
               --static_map ../static_maps/land_shapes_ht_crop.png
```