
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

## 예시 
<예시1 - 일생대화>  
<img width="500" alt="image" src="https://user-images.githubusercontent.com/73736988/230777095-7821d6e8-9f43-453d-8325-aa8abbb72b0f.png">  
  
  
<예시2 - 일생대화>  
<img width="500" alt="image" src="https://user-images.githubusercontent.com/73736988/230777372-b959c118-528b-49fe-8ddc-ffa68979a14a.png">  
  
  
<예시3 - 일생대화>  
<img width="500" alt="image" src="https://user-images.githubusercontent.com/73736988/230777306-b537ff16-7293-40d4-a31e-63b36eac434d.png">  
