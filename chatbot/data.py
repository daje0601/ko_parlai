import os
import pickle
from itertools import chain
from tqdm.auto import tqdm
from torch.utils.data import Dataset

from processing import Processing
 
class Dialogues(Processing):
    """
    여러 종류의 데이터를 처리하기 위해 데이터 클래스
    데이터를 토큰화하여 pickle로 저장합니다. 
    """
    def __init__(self, tokenizer, args):
        
        self.tokenizer = tokenizer
        self.args = args
        # self.dataset_list = ['daily_dialog', 'empathetic_dialogues', 'persona_chat', 'blended_skill_talk']
        self.dataset_list = ['topic_text_daily_dialog_AIhub', 'AIhub_korea_sns', 'modu_corpus_NIKL']
        super().__init__(tokenizer, args['train_frac'])

    def load(self):
        train_dataset = []
        valid_dataset = []

        for dataset_name in self.dataset_list:
            print(f'Loading {dataset_name} dataset...')

            train_dialogues, valid_dialogues = self._load_dialog(dataset=dataset_name)
            train_dataset += train_dialogues
            valid_dataset += valid_dialogues

        return train_dataset, valid_dataset

    def save(self, prefix, tokenizer, dialogues):
        print(f'Saving {prefix} dialogues to file...')

        if not os.path.isdir(self.args["dataset_dir"]):
            os.makedirs(self.args["dataset_dir"])

        dialogues_path = f'{self.args["dataset_dir"]}/{prefix}_dialogues.pickle'
        ids_path = f'{self.args["dataset_dir"]}/{prefix}_ids.pickle'

        with open(dialogues_path, 'wb') as f:
            pickle.dump(dialogues, f)

        print(f'Saving {prefix} ids to file...')
        ids = []
        for dialogue in tqdm(dialogues):
            dialogue_ids = []
            for utter in dialogue:
                token_ids = tokenizer.encode(utter)
                dialogue_ids.append(token_ids)
            ids.append(dialogue_ids)

        with open(ids_path, 'wb') as f:
            pickle.dump(ids, f)

        print('Saving complete!')

    def _load_dialog(self, dataset=None):
        if dataset == 'daily_dialog':
            return self._load_daily()
        elif dataset == 'empathetic_dialogues':
            return self._load_empathetic()
        elif dataset == 'persona_chat':
            return self._load_persona()
        elif dataset == 'blended_skill_talk':
            return self._load_blended()
        elif dataset == "topic_text_daily_dialog_AIhub":
            return self._load_topic_text_daily_dialog_AIhub()
        elif dataset == "AIhub_korea_sns":
            return self._load_AIhub_korea_sns_non_label()
        elif dataset == "modu_corpus_NIKL":
            return self._load_modu_corpus_NIKL()



class DialoguesDataset(Dataset):
    """
    저장했던 pickle 데이터를 load하여 대화 데이터셋으로 데이터 구조를 변경하고, special token을 추가합니다. 
    """
    def __init__(self, prefix, args):
        self.input_ids = []
        self.token_type_ids = []
        self.labels = []
        self._prepare_data(prefix, args)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.token_type_ids[idx], self.labels[idx]

    def _prepare_data(self, prefix, args):
        with open(f'{args["dataset_dir"]}/{prefix}_ids.pickle', 'rb') as f:
            dials = pickle.load(f)

        for dial in tqdm(dials): # dials은 화자들이 번갈아가며 이야기한 대화 리스트가 들어오게 됩니다. # 예시 : ['너희 집 근처에는 병원 있어?', '응, 난 엄청 근처에 있어.', '집 근처에 병원이 있으면 편리한 거 같아.', '왜 그렇게 생각하는 거야?', '사고가 나면 병원을 빨리 가는 게 중요해.', '나는 집 근처에 병원이 있어서 다행이야.', '다음에 이사 가면 근처에 병원 있는지 꼭 확인해야겠어.']
            hists = []
            for i, sentence in enumerate(dial):
                if i % 2 == 0:
                    hists.append([args['sp1_id']] + sentence)
                else:
                    hists.append([args['sp2_id']] + sentence)

            for i in range(len(hists)):
                if hists[i][0] == args['sp2_id']:
                    for j in range(0, i):
                        contexts = hists[j:i + 1]
                        if len(contexts) > args['max_history']:
                            num_exceeded = len(contexts) - args['max_history']
                            contexts = contexts[num_exceeded:]
                        if len(contexts) < 2:
                            break

                        input_ids = [args['bos_id']] + list(chain.from_iterable(contexts)) + [args['eos_id']]
                        if len(input_ids) <= args['max_len']:
                            start_sp_id, next_sp_id = contexts[0][0], contexts[1][0]
                            token_type_ids = [[start_sp_id] * len(ctx) if c % 2 == 0 else [next_sp_id] * len(ctx) for c, ctx in enumerate(contexts)]
                            token_type_ids = [start_sp_id] + list(chain.from_iterable(token_type_ids)) + [args['sp2_id']]

                            labels = [[-100] * len(ctx) if c < len(contexts) - 1 else [-100] + ctx[1:] for c, ctx in enumerate(contexts)]
                            labels = [-100] + list(chain.from_iterable(labels)) + [args['eos_id']]

                            self.input_ids.append(input_ids)
                            self.token_type_ids.append(token_type_ids)
                            self.labels.append(labels)

                            break

        del dials
