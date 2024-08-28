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

## Code interpolation
### Original Code
```ruby
%cd /content
```
##### Explanation: 
Easy to understand, just switch to the directory we're now working at to the directory called _content_ in your terminal. Should have no problem with this line. Let's keep moving!

### Original Code
```ruby
!git clone --recursive https://github.com/camenduru/gaussian-splatting
```
##### Explanation: 
Use git command to clone a repository from GitHub pages. If you do not know what Git is, it's a version control system developed for easier version control of files on Linux system, for more details, check the offcial website of [git](https://git-scm.com/docs/user-manual.html) Here, ```git clone --recursive <url> ``` is equal to ``` git clone --recurse-submodules <url>```, to initialize and update the submodules inside the project automatically. Details can refer to [git Documentation clone submodules](https://git-scm.com/docs/git-clone#Documentation/git-clone.txt-code--recurse-submodulescodecodecodeemltpathspecgtem) and [stackoverflow git clone submodules](https://stackoverflow.com/questions/3796927/how-do-i-git-clone-a-repo-including-its-submodules)
##### Notice:
If you ran this line on the colab page before, there might be a problem showing as: 
> destination path already exists and is not an empty directory 

This is because there is already a directory called _gaussian-splatting_ in your working directory, you can either remove it by ```rm -rf gaussian-splatting``` and rerun the command or simply skip this line of code.

### Original Code
```ruby
!pip install -q plyfile
```
#### Explanation:
use pip package installer for python to install the _pylfile_ module. ```pip install -q``` stands for [quiet install](https://pip.pypa.io/en/stable/cli/pip/#cmdoption-q). No problem will be encountered if you have the pip package installed, else install one.

### Original Code
```ruby
%cd /content/gaussian-splatting
```
#### Explanation:
get to the directory called _guassian-splatting_

### Original Code
```ruby
!pip install -q /content/gaussian-splatting/submodules/diff-gaussian-rasterization
```
#### Explanation:
This line is to use pip to install a project in a directory already downloaded on the computer, [local project install use pip](https://pip.pypa.io/en/stable/topics/local-project-installs/).  

If you check out your directory using the ```%cd /submodules``` command at this step, you will see two directories, one is _diff-guassian-rasterization_ and the other one is _simple-knn_. 
Look deeper into _diff-gaussian-rasterization_, you can see there is a file called _setup.py_, which contains the information of installation process, specify dependencies of the project files. 

Then let's read the code in _setup.py_:
```ruby
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
os.path.dirname(os.path.abspath(__file__))

setup(
    name="diff_gaussian_rasterization",
    packages=['diff_gaussian_rasterization'],
    ext_modules=[
        CUDAExtension(
            name="diff_gaussian_rasterization._C",
            sources=[
            "cuda_rasterizer/rasterizer_impl.cu",
            "cuda_rasterizer/forward.cu",
            "cuda_rasterizer/backward.cu",
            "rasterize_points.cu",
            "ext.cpp"],
            extra_compile_args={"nvcc": ["-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/")]})
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
```
A library called _setuptools_ is been included. Using with the file _pyproject.toml_, this library helps with [building and distributing packages of the project](https://www.bilibili.com/video/BV1y64y1U7cJ/?spm_id_from=333.337.search-card.all.click&vd_source=02a0a629234ac89b2b67c57092a6dada) and installer during installing the package. 



