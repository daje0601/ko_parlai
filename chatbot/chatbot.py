import yaml
import torch
import nltk
from glob import glob
from argparse import ArgumentParser
import os
os.environ["TOKENIZERS_PARALLELISM"] = "True"
from transformers import AutoModelForCausalLM, AutoTokenizer

from data import Dialogues
from utils import set_seed

import gradio as gr

def main(args):
    
    
    set_seed(args['seed'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    args['device'] = device

    tokenizer = load_tokenizer(args)
    model = load_model(args, tokenizer, device)

    if dataset_is_missing(args):
        dialogues = Dialogues(tokenizer, args)
        train_dataset, valid_dataset = dialogues.load()

        dataset_types = ['train', 'valid']
        datasets = [train_dataset, valid_dataset]

        for dataset_type, dataset in zip(dataset_types, datasets):
            dialogues.save(dataset_type, tokenizer, dataset)

    if args['mode'] == 'train':
        from train import Trainer
        trainer = Trainer(model, args)
        trainer.train()
    elif args['mode'] == 'interact':
        from interact import Chatbot
        chatbot = Chatbot(model, tokenizer, args)
        chatbot.run()
    elif args['mode'] == 'gradio':
        from itertools import chain
        import torch.nn.functional as F
        from utils import top_k_filter, lemma_sentence
        path = args['checkpoint']
        checkpoint = torch.load(path, map_location=args['device'])
        model.load_state_dict(checkpoint['model_state_dict'])

        def _top_filtering(input_ids, token_type_ids):
            
            output_ids = []

            for pos in range(args['max_len']):
                # output = model(input_ids=input_ids, token_type_ids=token_type_ids)[0]
                output = model(input_ids=input_ids)[0]

                logits = output[0, -1, :] / args['temperature']
                logits = top_k_filter(logits, top_k=args['top_k'])
                output = F.softmax(logits, dim=-1).unsqueeze(0)

                sorted_probs, sorted_idxs = torch.sort(output, descending=True)
                cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
                idx_remove = cumsum_probs > args['top_p']
                idx_remove[:, 1:] = idx_remove[:, :-1].clone()
                idx_remove[:, 0] = False
                sorted_probs[idx_remove] = 0.0
                sorted_probs /= torch.sum(sorted_probs, dim=-1, keepdim=True)

                probs = torch.zeros(output.shape, device=args['device']).scatter_(-1, sorted_idxs, sorted_probs)
                idx = torch.multinomial(probs, 1)

                idx_item = idx.squeeze(-1).squeeze(-1).item()

                if idx_item in output_ids:
                    continue

                output_ids.append(idx_item)

                if idx_item == args['eos_id']:
                    break

                input_ids = torch.cat((input_ids, idx.reshape(1, 1)), dim=-1)
                next_type_id = torch.LongTensor([[args['sp2_id']]]).to(args['device'])
                token_type_ids = torch.cat((token_type_ids, next_type_id), dim=-1)
                assert input_ids.shape == token_type_ids.shape

            return output_ids

        
        def gradio_prediction(inputs, history=[]):
            history.append(inputs)
            
            input_hists = []

            input_ids = [args['sp1_id']] + tokenizer.encode(inputs)
            input_hists.append(input_ids)

            if len(input_hists) >= args['max_history']:
                num_exceeded = len(input_hists) - args['max_history']
                input_hists = input_hists[num_exceeded:]

            input_ids = [args['bos_id']] + list(chain.from_iterable(input_hists)) + [args['sp2_id']]
            start_sp_id = input_hists[0][0]
            next_sp_id = args['sp1_id'] if start_sp_id == args['sp2_id'] else args['sp2_id']
            token_type_ids = [[start_sp_id] * len(hist) if h % 2 == 0 else [next_sp_id] * len(hist) for h, hist in enumerate(input_hists)]
            assert len(token_type_ids) == len(input_hists)
            token_type_ids = [start_sp_id] + list(chain.from_iterable(input_hists)) + [args['sp2_id']]
            assert len(input_ids) == len(token_type_ids)
            input_len = len(input_ids)

            input_ids = torch.LongTensor(input_ids).unsqueeze(0).to(args['device'])
            token_type_ids = torch.LongTensor(token_type_ids).unsqueeze(0).to(args['device'])

            output_ids = _top_filtering(input_ids, token_type_ids)
            answer = tokenizer.decode( output_ids, 
                                            num_beams=5, 
                                            top_k=20,
                                            no_repeat_ngram_size=4, 
                                            length_penalty=0.55,
                                            repetition_penalty=2.0,
                                            skip_special_tokens=True)

            print(f'Bot: {answer}')
            history.append(answer)
            input_hists.append([args['sp2_id']] + tokenizer.encode(answer))
            response = [(history[i], history[i+1]) for i in range(0, len(history)-1, 2)]
            return response, history 

        gr.Interface(fn=gradio_prediction,
             inputs=["text", "state"],
             outputs=["chatbot", "state"]).launch(share=True)

        # block = gr.Blocks()
        # with block:
        #     gr.Markdown("자유롭게 이야기 해보세요.")
        #     with gr.Row():
        #         display = gr.outputs.Chatbot()
        #     with gr.Row():
        #         text1 = gr.inputs.Textbox()
        #     with gr.Row():
        #         button1 = gr.Button(label="Chat")
        #     button1.click(gradio_prediction, text1, display)
        # block.launch()



def dataset_is_missing(args):
    if len(glob(f'{args["dataset_dir"]}/*.pickle')) == 0:
        return True
    return False


def load_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args['model_name'])
    special_tokens = ['<speaker1>', '<speaker2>', '<bos>', '<eos>']
    tokenizer.add_special_tokens({
        'bos_token': '<bos>',
        'eos_token': '<eos>',
        'additional_special_tokens': special_tokens
    })
    # add new token ids to args
    sp1_id, sp2_id, bos_id, eos_id = tokenizer.convert_tokens_to_ids(special_tokens)
    args['sp1_id'] = sp1_id
    args['sp2_id'] = sp2_id
    args['bos_id'] = bos_id
    args['eos_id'] = eos_id
    return tokenizer

def load_model(args, tokenizer, device):
    model = AutoModelForCausalLM.from_pretrained(args['model_name']).to(device)
    model.resize_token_embeddings(len(tokenizer))
    return model


if __name__ == '__main__':
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    nltk.download('punkt')

    parser = ArgumentParser()
    parser.add_argument('--mode', type=str, required=True,
                        help='Pass "train" for training mode and "interact" for interaction mode')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint of the model')

    user_args = parser.parse_args()
    arguments = yaml.safe_load(open('config.yml'))
    arguments.update(vars(user_args))

    main(arguments)
