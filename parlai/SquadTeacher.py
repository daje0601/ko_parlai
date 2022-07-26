from parlai.core.teachers import register_teacher, DialogTeacher
from parlai.scripts.display_model import DisplayModel

@register_teacher("my_teacher")
class SquadTeacher(DialogTeacher):
    def __init__(self, opt, shared=None):
        self.datatype = opt['datatype']
        build(opt)  # NOTE: the call to build here
        suffix = 'train' if opt['datatype'].startswith('train') else 'dev'
        # whatever is placed into datafile will be passed as the argument to
        # setup_data in the next section.
        opt['datafile'] = os.path.join(opt['datapath'], 'SQuAD', suffix + '-v1.1.json')
        self.id = 'squad'
        super().__init__(opt, shared)
    
    def setup_data(self, path):
        # note that path is the value provided by opt['datafile']
        print('loading: ' + path)
        with PathManager.open(path) as data_file:
            self.squad = json.load(data_file)['data']
        for article in self.squad:
            # each paragraph is a context for the attached questions
            for paragraph in article['paragraphs']:
                # each question is an example
                for qa in paragraph['qas']:
                    question = qa['question']
                    answers = tuple(a['text'] for a in qa['answers'])
                    context = paragraph['context']
                    yield {"text": content + "\n" + question, "labels": answers}, True


