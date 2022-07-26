import os 
import sys
import json
import torch 
from tqdm.auto import tqdm
sys.path.append("/Users/kds/Documents/openchat/workspace/ko_parlai/pororo")
from pororo import Pororo



mt = Pororo(task="mt", lang="multi", model="transformer.large.multi.mtpg")
path = "/opt/homebrew/Caskroom/miniforge/base/envs/parl/lib/python3.8/site-packages/data/wizard_of_interent/train.jsonl"

with open(path, "r") as file:
    _idx = 0
    for line in file:    
        if _idx > 1301 :
            break 
        elif _idx >= 1300:
                
            print(f"{_idx}/2600 번역중입니다. ")
            data = json.loads(line)
            
            result_dict = {}  # {"146580" : }형식으로 들어가야함
            except_list = []

            for main_key, _ in data.items():
                main_key = main_key  # 3842 etc.
            
                sub_dict = {}
                for key, value in data[main_key].items():  # key :apprentice_persona, dialog_history, start_timestamp
                    if key == "apprentice_persona":
                        a_translation = mt(value, src="en", tgt="ko")
                        sub_dict[key] = a_translation
                        

                    elif key == "dialog_history":
                        dialog_total_list = []
                        
                        for i, value_dict in enumerate(value): # type(value)== "list"
                            
                            dialog_dict = {}
                            for d_key, d_value in value_dict.items(): #type(value_dict) == "dict"
                                if d_key == "action":
                                    dialog_dict[d_key] = d_value

                                elif d_key == "text":
                                    if len(d_value) > 0:
                                        d_t_translation = mt(d_value, src="en", tgt="ko")
                                        dialog_dict[d_key] = d_t_translation
                                    elif d_value == "":
                                        dialog_dict[d_key] = d_value

                                elif d_key == "context":
                                    context_dict = {}
                                    if len(d_value) == 0:
                                        dialog_dict[d_key] = d_value
                                    else:
                                        for d_c_key,d_c_value in d_value.items():  # d_value == dict("contents"(list)) 
                                            if d_c_key == "contents":# d_c_key == "contents"
                                                contents_list = []

                                                for j, d_c_c_value in enumerate(d_c_value):  # type(d_c_value) == list(이 리스트는 dict로 구성되어 있음)
                                                    
                                                    contents_sub_dict = {}
                                        
                                                    for d_c_c_c_key, d_c_c_c_value in d_c_c_value.items():# d_c_c_value == dict(url(str), title(str), content(list)" -> dict가 하나씩 들어옴
                                                        if d_c_c_c_key == "url":
                                                            contents_sub_dict[d_c_c_c_key] = d_c_c_c_value
                                                        elif  d_c_c_c_key == "title":
                                                            contents_sub_dict[d_c_c_c_key] = mt(d_c_c_c_value, src="en", tgt="ko")
                                                        elif d_c_c_c_key == "content":
                                                            contents_sub_contnet_list = []
                                                            for one_sentence_in_content in d_c_c_c_value:
                                                                try:
                                                                    d_c_c_c_value_result = mt(one_sentence_in_content, src="en", tgt="ko")
                                                                    contents_sub_contnet_list.append(d_c_c_c_value_result)
                                                                    
                                                                except:
                                                                    print(f"{main_key}의 dialogue {i}번의 contents {j} dict가 예외 처리가 되었습니다.",)
                                                                    except_list.append([main_key,i,j])
                                                                    contents_sub_contnet_list.append(one_sentence_in_content)
                                                            contents_sub_dict[d_c_c_c_key] = contents_sub_contnet_list
                                                        
                                                    contents_list.append(contents_sub_dict)

                                                context_dict[d_c_key] = contents_list
                                                dialog_dict[d_key] = context_dict

                                            elif d_c_key == "selected_contents":
                                                context_dict[d_c_key] = d_c_value
                                                dialog_dict[d_key] = context_dict
                                    

                                elif d_key == "timestamp":
                                    dialog_dict[d_key] = d_value
                            dialog_total_list.append(dialog_dict)
                        sub_dict[key] = dialog_total_list  # "dialog_history"

                    elif key == "start_timestamp":
                        sub_dict[key] = value

                result_dict[main_key] = sub_dict


            with open(os.path.join("/Users/kds/Documents/openchat/workspace/ko_parlai/translation_dataset/translation_wizard_of_internet",
                    "translation_wizard_of_Internet_train1300_2600.jsonl",), "a", encoding="UTF-8",) as f:
                f.write(json.dumps(result_dict, ensure_ascii=False) + "\n")
            with open(os.path.join("/Users/kds/Documents/openchat/workspace/ko_parlai/translation_dataset/translation_wizard_of_internet",
                        "translation_wizard_of_Internet_train1300_2600_exception_list.txt",), "a", encoding="UTF-8",) as f:
                    f.write(str(except_list) + "\n")    
        _idx += 1
