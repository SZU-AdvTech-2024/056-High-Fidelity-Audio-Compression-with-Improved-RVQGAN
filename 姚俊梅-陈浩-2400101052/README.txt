1、由于github的源代码不符合我们自己的数据集结果，所以我将dataset重构用于适应我们自己的原始数据集；
2、作者提供的train.py代码不符合我们的预期，所以我将作者提供的训练代码进行了自我编写，符合了我自己的训练需求；

1、使用Descript Audio Codec，可以将44.1 KHz 音频压缩为低 8 kbps 比特率的离散代码。
2、这大约是90 倍的压缩，同时保持出色的保真度并最大限度地减少伪影。
3、模型适用于所有领域（语音、环境、音乐等），使其广泛应用于所有音频的生成建模。
4、它可以用作所有音频语言建模应用程序（例如 AudioLMs、MusicLMs、MusicGen 等）的 EnCodec 的替代品。

用法
安装：pip install descript-audio-codec
或者是：pip install git+https://github.com/descriptinc/descript-audio-codec

单GPU：
export CUDA_VISIBLE_DEVICES=0
python scripts/dtrain.py 

环境安装：
pip install requirements.txt

代码只要环境配置好，可以直接运行。我们的数据集由于过于庞大，我在dataset-download.txt给与下载地址。


