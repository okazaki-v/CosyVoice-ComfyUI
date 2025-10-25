# CosyVoice-ComfyUI

一个 [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) 的 ComfyUI 自定义节点。

**项目已更新，现已同时支持 `CosyVoice V1` 和 `CosyVoice V2` 的功能。**

你可以在 `workflows` 目录中找到示例工作流。

## 如何使用

**1. 安装依赖**

请确保你的环境中已安装 `ffmpeg`。

- **Linux:**
  ```bash
  apt update && apt install ffmpeg
  ```
- **Windows:**
  推荐使用 [WingetUI](https://github.com/marticliment/WingetUI) 自动安装 `ffmpeg`。

**2. 安装节点**

```bash
# 进入 ComfyUI/custom_nodes/ 目录
git clone https://github.com/AIFSH/CosyVoice-ComfyUI.git
cd CosyVoice-ComfyUI
pip install -r requirements.txt
```

**3. 下载模型**

**无需手动下载**。节点在第一次使用时，会自动从 ModelScope 下载所需的 V1 或 V2 模型。

## 功能介绍

### CosyVoiceNode (V1 & V2 TTS)

这是核心的文本到语音生成节点。你可以在 `inference_mode` 选项中选择使用 V1 或 V2 的不同功能。

#### CosyVoice V2 支持

新增支持 **CosyVoice V2** 模型 (`iic/CosyVoice2-0.5B`)，输出采样率为 **24kHz**。

- `v2/零样本语音克隆 (Zero-shot)`: 使用一小段参考音频和对应文本，克隆音色。
  - **需要**: `tts_text`, `prompt_text`, `prompt_wav`
- `v2/跨语种语音克隆 (Cross-lingual)`: 使用参考音频，跨语种克隆音色。
  - **需要**: `tts_text`, `prompt_wav`
- `v2/指令控制合成 (Instruct)`: 使用指令和参考音频，控制合成风格。
  - **需要**: `tts_text`, `instruct_text`, `prompt_wav`

#### CosyVoice V1 支持

继续支持 **CosyVoice V1** (`iic/CosyVoice-300M`) 的全部功能，输出采样率为 **22.05kHz**。

- `v1/预训练音色 (SFT)`: 使用预设的音色进行合成。
- `v1/3s极速复刻 (Zero-shot)`: 使用3秒音频和文本进行音色克隆。
- `v1/跨语种复刻 (Cross-lingual)`: 跨语种进行音色克隆。
- `v1/自然语言控制 (Instruct)`: 使用自然语言指令控制合成风格。

### CosyVoiceDubbingNode (SRT配音)

此节点（目前基于V1模型）支持根据 `srt` 字幕文件对音频进行配音，可实现单人或多人音色克隆。

- **输入**:
  - `tts_srt`: 目标语言的字幕文件。
  - `prompt_wav`: 包含参考音色的完整音频。
  - `prompt_srt` (可选): 参考音频对应的字幕文件。

## 教程

- [Bilibili视频演示](https://www.bilibili.com/video/BV16H4y1w7su)