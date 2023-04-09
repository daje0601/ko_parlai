# ☕ GPT2 Chatbot

GPT-2 chatbot for daily conversations trained on `Daily Dialogue`, `Empathetic Dialogues`, `PERSONA-CHAT`, `Blended Skill Talk` datasets. This chatbot is made based on GPT2 Model transformer with a language modeling head on top.

![chatbot](https://user-images.githubusercontent.com/70326958/151570518-ce70261a-6e8e-47a0-92e5-2d7638e7aa68.jpg)


## 설치 방법
```sh
git clone https://github.com/daje0601/ko_parlai.git
conda create -n ko_parlai python=3.8 -y 
conda activate ko_parlai 
pip install -r requirements.txt
```

## Train 방법 
```sh
python chatbot.py --mode train
```

## Interaction 방법 
```sh
python chatbot.py --mode interact --checkpoint path/to/model.h5
```

## gradio 방법 
```sh
python chatbot.py --mode gradio --checkpoint path/to/model.h5
```