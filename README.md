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
