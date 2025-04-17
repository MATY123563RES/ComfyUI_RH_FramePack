# FramePack for ComfyUI

## Online Access
You can also access RunningHub online to use this plugin and model for free.
Run&Download this Workflow: 
https://www.runninghub.ai/post/1912930457355517954

## Features  
This is a simple implementation of https://github.com/lllyasviel/FramePack. If there are any advantages, they would be:  
- Better automatic adaptation for 24GB GPUs, enabling higher resolution processing whenever possible.  
- The entire workflow requires no parameter adjustments, making it extremely user-friendly.  

## Model Configuration in the models dir:
- hunyuan_root = '/workspace/comfyui/models/HunyuanVideo'
- flux_redux_bfl_root = '/workspace/comfyui/models/flux_redux_bfl'
- framePackI2V_root = '/workspace/comfyui/models/FramePackI2V_HY'
```
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

