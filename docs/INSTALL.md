# Installation

1. Clone and enter this repository:
    ```
    git clone git@github.com:timmeinhardt/trackformer.git
    cd trackformer
    ```

2. Install packages for Python 3.7:

    1. `pip3 install -r requirements.txt`
    2. Install PyTorch 1.5 and torchvision 0.6 from [here](https://pytorch.org/get-started/previous-versions/#v150).
    3. Install pycocotools (with fixed ignore flag): `pip3 install -U 'git+https://github.com/timmeinhardt/cocoapi.git#subdirectory=PythonAPI'`
    5. Install MultiScaleDeformableAttention package: `python src/trackformer/models/ops/setup.py build --build-base=src/trackformer/models/ops/ install`

3. Download and unpack datasets in the `data` directory:

    

    [HT21] (https://motchallenge.net/data/HT21/):
       ```
       wget https://motchallenge.net/data/HT21.zip
       unzip HT21.zip
       python3 src/generate_coco_from_HT21.py
       ```    
    

3. Download and unpack pretrained TrackFormer model files which is the starting point model in this project in the `models` directory:

    ```
    wget https://vision.in.tum.de/webshare/u/meinhard/trackformer_models_v1.zip
    unzip trackformer_models_v1.zip
    ```

