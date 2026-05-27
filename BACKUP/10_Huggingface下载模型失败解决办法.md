# [Huggingface下载模型失败解决办法](https://github.com/ruheyun/yc-blog/issues/10)

## 正常下载模型方法

```
from huggingface_hub import snapshot_download

snapshot_download("Qwen/Qwen3-0.6B")
```

但是由于网络问题，无法直接从Huggingface下载模型文件到本地。
可先清除缓存，尝试重新下载。仍然可能失败。

```
rm -rf ~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B
```

## 解决办法

### 设置镜像网站

```
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
```

在需要下载模型的python脚本最上方前两行插入上述代码，这样可能会解决网络问题。但仍然可能有模型找不到的情况，或者镜像返回的 metadata 不符合官方接口格式而报错。

### 使用git下载模型

首先确保安装了git和git-lfs，下面是Ubuntu系统下安装命令。

```
sudo apt update
sudo apt install git
sudo apt install git-lfs
```

Huggingface的模型文件是基于git进行远程管理的，因此可以通过克隆来下载模型。找到模型页面，复杂上方地址栏里的地址，就是仓库地址。

```
git clone https://huggingface.co/Qwen/Qwen3-0.6B
```

一般模型权重较大，通过clone下载的都是Git LFS pointer文件（指针文件），并不是真实权重。进行模型文件夹，安装模型。

```
cd Qwen3-0.6B
git lfs pull
```

之后使用本地路径即可。

### 手动下载

找到Huggingface模型页面，点击‘Files and versions’后下载页面所有文件，放到本地文件夹Qwen3-0.6B（模型的名字），然后使用本地路径。