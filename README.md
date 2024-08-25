# 3d-GS-Reproduction-problems
To record the problems I met in reproduction 3d Gaussian Splatting on Google Colab without paying credits.

### Original Colab link is here: [3D Gaussian Splatting Original Colab](https://colab.research.google.com/github/camenduru/gaussian-splatting-colab/blob/main/gaussian_splatting_colab.ipynb)

#### The full code is here: 
```ruby
%cd /content
!git clone --recursive https://github.com/camenduru/gaussian-splatting
!pip install -q plyfile

%cd /content/gaussian-splatting
!pip install -q /content/gaussian-splatting/submodules/diff-gaussian-rasterization
!pip install -q /content/gaussian-splatting/submodules/simple-knn

!wget https://huggingface.co/camenduru/gaussian-splatting/resolve/main/tandt_db.zip
!unzip tandt_db.zip

!python train.py -s /content/gaussian-splatting/tandt/train

# !wget https://huggingface.co/camenduru/gaussian-splatting/resolve/main/GaussianViewTest.zip
# !unzip GaussianViewTest.zip
# !python render.py -m /content/gaussian-splatting/GaussianViewTest/model
# !ffmpeg -framerate 3 -i /content/gaussian-splatting/GaussianViewTest/model/train/ours_30000/renders/%05d.png -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -c:v libx264 -r 3 -pix_fmt yuv420p /content/renders.mp4
# !ffmpeg -framerate 3 -i /content/gaussian-splatting/GaussianViewTest/model/train/ours_30000/gt/%05d.png -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -c:v libx264 -r 3 -pix_fmt yuv420p /content/gt.mp4 -y
```

Seems to be quite short and easy to implement Huh? When I first tried to implement it on Google Colab, the process was extremely easy and smooth. Just a few hours the training process of 30k iterations got done, and I stopped at the step of visualization. However, several weeks later, I want to try it again, a lot of problems appeared. 
I will try to write down the problems I met and how I solved them. Links of blogs which I referred to will also be listed (if I remembered them all).
