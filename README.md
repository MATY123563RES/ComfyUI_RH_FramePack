# FramePack for ComfyUI

## Online Access
You can access RunningHub online to use this plugin and models for free:
### English Version
- **Run & Download Workflow**:  
  [https://www.runninghub.ai/post/1912930457355517954](https://www.runninghub.ai/post/1912930457355517954)
### 中文版本
- **运行并下载工作流**:  
  [https://www.runninghub.cn/post/1912930457355517954](https://www.runninghub.cn/post/1912930457355517954)

## Features  
This is a simple implementation of https://github.com/lllyasviel/FramePack. If there are any advantages, they would be:  
- Better automatic adaptation for 24GB GPUs, enabling higher resolution processing whenever possible.  
- The entire workflow requires no parameter adjustments, making it extremely user-friendly.  

## Model Configuration in the `models` Directory
- **国内用户网盘下载**:  [汤团猪大佬的模型整合包] (https://pan.baidu.com/s/18KM-yPJL-nRhHBRjPpR_PQ?pwd=rx89) 
### 1. HunyuanVideo Model
- **Local Path**: `/workspace/comfyui/models/HunyuanVideo`
- **Download Source**: [HunyuanVideo on HuggingFace](https://huggingface.co/hunyuanvideo-community/HunyuanVideo/tree/main)

### 2. Flux Redux BFL Model
- **Local Path**: `/workspace/comfyui/models/flux_redux_bfl`
- **Download Source**: [flux_redux_bfl on HuggingFace](https://huggingface.co/lllyasviel/flux_redux_bfl/tree/main)

### 3. FramePackI2V Model
- **Local Path**: `/workspace/comfyui/models/FramePackI2V_HY`
- **Download Source**: [FramePackI2V_HY on HuggingFace](https://huggingface.co/lllyasviel/FramePackI2V_HY/tree/main)
```
comfyui/models/
  flux_redux_bfl
  ├── feature_extractor
  │   └── preprocessor_config.json
  ├── image_embedder
  │   ├── config.json
  │   └── diffusion_pytorch_model.safetensors
  ├── image_encoder
  │   ├── config.json
  │   └── model.safetensors
  ├── model_index.json
  └── README.md
  FramePackI2V_HY
  ├── config.json
  ├── diffusion_pytorch_model-00001-of-00003.safetensors
  ├── diffusion_pytorch_model-00002-of-00003.safetensors
  ├── diffusion_pytorch_model-00003-of-00003.safetensors
  ├── diffusion_pytorch_model.safetensors.index.json
  └── README.md
  HunyuanVideo
  ├── config.json
  ├── model_index.json
  ├── README.md
  ├── scheduler
  │   └── scheduler_config.json
  ├── text_encoder
  │   ├── config.json
  │   ├── model-00001-of-00004.safetensors
  │   ├── model-00002-of-00004.safetensors
  │   ├── model-00003-of-00004.safetensors
  │   ├── model-00004-of-00004.safetensors
  │   └── model.safetensors.index.json
  ├── text_encoder_2
  │   ├── config.json
  │   └── model.safetensors
  ├── tokenizer
  │   ├── special_tokens_map.json
  │   ├── tokenizer_config.json
  │   └── tokenizer.json
  ├── tokenizer_2
  │   ├── merges.txt
  │   ├── special_tokens_map.json
  │   ├── tokenizer_config.json
  │   └── vocab.json
  ├── transformer
  │   ├── config.json
  │   ├── diffusion_pytorch_model-00001-of-00006.safetensors
  │   ├── diffusion_pytorch_model-00002-of-00006.safetensors
  │   ├── diffusion_pytorch_model-00003-of-00006.safetensors
  │   ├── diffusion_pytorch_model-00004-of-00006.safetensors
  │   ├── diffusion_pytorch_model-00005-of-00006.safetensors
  │   ├── diffusion_pytorch_model-00006-of-00006.safetensors
  │   └── diffusion_pytorch_model.safetensors.index.json
  └── vae
      ├── config.json
      └── diffusion_pytorch_model.safetensors
```
## Example:
![image](https://github.com/user-attachments/assets/ea936caf-c0ca-48f4-af20-64090771d382)

