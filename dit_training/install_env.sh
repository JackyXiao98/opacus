conda create -n dit python=3.10

conda activate dit

conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia

pip install timm diffusers accelerate

rm -rf build/ dist/ opacus.egg-info/ .eggs/ PKG-INFO

rm -rf ~/.cache/python_metadata/



