import ast
import json
from tqdm.auto import tqdm



class Processing:
    def __init__(self, tokenizer, train_frac):
        self.tokenizer = tokenizer
        self.train_frac = train_frac

    def _load_persona(self):
        dataset_path = "/home/kds/workspace/Ko_history_dataset.json"
        with open(dataset_path, 'r') as j:
            dataset = json.loads(j.read())

        train_data = dataset['train']["Ko"]
        valid_data = dataset['valid']["Ko"]
        all_data = train_data + valid_data
        all_dialogues = []

        for obj in tqdm(all_data):
            dialogue = obj['history']
            new_dialogue = []

            for i, sentence in enumerate(dialogue):
                if sentence.strip() != '__ SILENCE __':
                    token_list = self.tokenizer.tokenize(sentence.strip())
                    text = self.tokenizer.convert_tokens_to_string(token_list)
                    new_dialogue.append(text)

            all_dialogues.append(new_dialogue)

        train_dialogues = all_dialogues[:int(len(all_dialogues) * self.train_frac)]
        valid_dialogues = all_dialogues[int(len(all_dialogues) * self.train_frac):]

        return train_dialogues, valid_dialogues


    def _load_AIhub_korea_sns(self):
        train_dataset_path = "/home/kds/workspace/Datasets_AIHub/korea_sns/Training/preprocessed_data/AIHub_sns_preprocessed_all_concat_train_data.json"
        valid_dataset_path = "/home/kds/workspace/Datasets_AIHub/korea_sns/Validation/preprocessed_data/AIHub_sns_preprocessed_all_concat_valid_data.json"
        
        with open(train_dataset_path, 'r') as train_dataset_files:
            train_dataset = json.loads(train_dataset_files.read())
        
        with open(valid_dataset_path, 'r') as valid_dataset_files:
            valid_dataset = json.loads(valid_dataset_files.read())
        
        all_data = train_dataset["train"] + valid_dataset["valid"]
        
        all_dialogues = []

        for obj in tqdm(all_data):
            
            dialogue = ast.literal_eval(obj)["dialog"]
            
            new_dialogue = []

            for i, sentence in enumerate(dialogue):
                token_list = self.tokenizer.tokenize(sentence[1].strip()) # 여기서는 tokneizer로 바꾸어주고 
                sentence[1] = self.tokenizer.convert_tokens_to_string(token_list) # 그 바꾼 것을 다시 글자로 변경을 해서 집어 넣네
                new_dialogue.append(sentence)
            
            all_dialogues.append(new_dialogue)
        
        train_dialogues = all_dialogues[:int(len(all_dialogues) * self.train_frac)]
        valid_dialogues = all_dialogues[int(len(all_dialogues) * self.train_frac):]

        return train_dialogues, valid_dialogues
    

    def _load_AIhub_korea_sns_non_label(self):
        train_dataset_path = "/home/kds/workspace/Datasets_AIHub/korea_sns/Training/train_non_labeld_data.json"
        valid_dataset_path = "/home/kds/workspace/Datasets_AIHub/korea_sns/Validation/valid_non_labeld_data.json"
        
        with open(train_dataset_path, 'r') as train_dataset_files:
            train_dataset = json.loads(train_dataset_files.read())
        
        with open(valid_dataset_path, 'r') as valid_dataset_files:
            valid_dataset = json.loads(valid_dataset_files.read())
        
        all_data = train_dataset["train"] + valid_dataset["valid"]
        
        all_dialogues = []

        for obj in tqdm(all_data):
            dialogue = ast.literal_eval(obj)["dialog"]
            
            new_dialogue = []

            for i, sentence in enumerate(dialogue):
                token_list = self.tokenizer.tokenize(sentence.strip()) # 여기서는 tokneizer로 바꾸어주고 
                text = self.tokenizer.convert_tokens_to_string(token_list) # 그 바꾼 것을 다시 글자로 변경을 해서 집어 넣네
                new_dialogue.append(text)
            
            all_dialogues.append(new_dialogue)
        
        train_dialogues = all_dialogues[:int(len(all_dialogues) * self.train_frac)]
        valid_dialogues = all_dialogues[int(len(all_dialogues) * self.train_frac):]

        return train_dialogues, valid_dialogues


    def _load_topic_text_daily_dialog_AIhub(self):
        train_dataset_path = "/home/kds/workspace/Datasets_AIHub/topic_text_daily_dialog_dataset/edit_dialogue/train_topic_text_daily_dialog.jsonl"
        with open(train_dataset_path, 'r') as train_dataset_files:
            train_dataset = json.loads(train_dataset_files.read())
        train_data = train_dataset["train"]
        all_data = train_data
        all_dialogues = []

        for obj in tqdm(all_data):
            dialogue = obj['dialog']
            new_dialogue = []

            for i, sentence in enumerate(dialogue):
                token_list = self.tokenizer.tokenize(sentence.strip())
                text = self.tokenizer.convert_tokens_to_string(token_list)
                new_dialogue.append(text)
            
            all_dialogues.append(new_dialogue)
        
        train_dialogues = all_dialogues[:int(len(all_dialogues) * self.train_frac)]
        valid_dialogues = all_dialogues[int(len(all_dialogues) * self.train_frac):]

        return train_dialogues, valid_dialogues
    

    def _load_modu_corpus_NIKL(self):
        train_dataset_path = "/home/kds/workspace/modu_NIKL.json"
        with open(train_dataset_path, 'r') as train_dataset_files:
            all_data = json.loads(train_dataset_files.read())
        
        all_dialogues = []

        for dialogue in tqdm(all_data['dialog']):

            new_dialogue = []

            for i, sentence in enumerate(dialogue):
                token_list = self.tokenizer.tokenize(sentence.strip())
                text = self.tokenizer.convert_tokens_to_string(token_list)
                new_dialogue.append(text)
            
            all_dialogues.append(new_dialogue)
        
        train_dialogues = all_dialogues[:int(len(all_dialogues) * self.train_frac)]
        valid_dialogues = all_dialogues[int(len(all_dialogues) * self.train_frac):]

        return train_dialogues, valid_dialogues


