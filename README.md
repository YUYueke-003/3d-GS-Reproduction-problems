# 3d-GS-Reproduction-problems
To record the problems I met in reproduction 3d Gaussian Splatting on Google Colab without paying credits. (OK, it's inpractical, but I will not delete it! I will try to figure out what the code is doing... Just forget about the viewer thing) 

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
__Analysis:__ ```from setuptools import setup```

A library called [_setuptools_](https://setuptools.pypa.io/en/latest/userguide/#building-and-distributing-packages-with-setuptools) has been included in the file. This library is an enhancement of the standard template library called [_distutils_](https://docs.python.org/zh-cn/3.10/library/distutils.html)(has been removed since python version 3.12, replaced by _setuptools_ and _packaging_). Using with the file _pyproject.toml_, this library helps with building and distributing packages of the project and installer during installing the package. More details please refer to building and [packaging](https://zhuanlan.zhihu.com/p/276461821).

__Analysis:__ ```from torch.utils.cpp_extension import CUDAExtension, BuildExtension```

This line is creating a custom class _setuptools.extension_ for CUDA and Build with bare minimum arguments, and all the arguments defined later (in the bracket starting with setup) will be forwarded to the class _setuptools.extension_ constructor. Details refer to this [page](https://pytorch.org/docs/stable/cpp_extension.html#torch-utils-cpp-extension).

__Analysis:__
```
import os
os.path.dirname(os.path.abspath(__file__))
```
_os_ is a STL in python, stands for operating system. The next line is to get a path of a file, [```os.path.dirname(path)```](https://docs.python.org/3/library/os.path.html#os.path.dirname) to get the parent path of a path and [```os.path.abspath```](https://docs.python.org/3/library/os.path.html#os.path.abspath) to get the absolute path of a file. So here, it is used to find the parent path of the absolute path of ```__file__```, equal to give the absolute path of the directory contains the ```__file__```.

