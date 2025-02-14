# Project_Chinese_Emotion_Classification
NLP - 中文句子情緒分類 Chinese sentence emotion classification

## Datasets Source 訓練資料來源
- [Datasets:Johnson8187/Chinese_Multi-Emotion_Dialogue_Dataset](https://huggingface.co/datasets/Johnson8187/Chinese_Multi-Emotion_Dialogue_Dataset)

## Base Model 基礎模型
- [hfl/chinese-roberta-wwm-ext](https://huggingface.co/hfl/chinese-roberta-wwm-ext)

## Project Structure
```

│
├─output
│  ├── config.json
│  ├── model.safetensors
│  ├── special_tokens_map.json
│  ├── tokenizer.json
│  ├── tokenizer_config.json
│  ├── training_args.bin
│  └── vocab.txt
│
│  Finetune_Chinese_emotion_classification.ipynb
│  Predict_Chinese_emotion_classification.ipynb
│  README.md
│

```

## GPU(Local Host) 
```
Fri Feb 14 11:18:15 2025       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 555.97                 Driver Version: 555.97         CUDA Version: 12.5     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                  Driver-Model | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 2070      WDDM  |   00000000:01:00.0 Off |                  N/A |
| N/A   41C    P0             38W /  115W |       0MiB /   8192MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
                                                                                         
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+

```

## Versioning
Python version at least 3.10

## Package 安裝套件
-torch                        2.6.0+cu124
-torchaudio                   2.6.0+cu124
-torchvision                  0.21.0+cu124
-transformers                 4.48.3
-scikit-learn                 1.6.1
-evaluate                     0.4.3
-accelerate                   1.3.0
-datasets                     3.2.0

## Description 說明
- 因Conda環境沒有cuda 12.5版的載點，故使用cuda 12.4版替代
- 使用roBERTa中文模型

## Result 成果
[![roBERTa_Chinese_Sentence_Emotion_Classification](https://img.youtube.com/vi/URBbeCM-yoY/0.jpg
)](https://youtu.be/URBbeCM-yoY)





