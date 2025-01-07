import torch
from sklearn.preprocessing import LabelEncoder

def get_device():
    if torch.cuda.is_available():
        print("device = cuda")
        return torch.device('cuda')
    else:
        print("device = cpu")
        return torch.device('cpu')

def add_tokens_to_tokenizer(args, tokenizer):
    # TODO : Replace with special tokens from SOLD dataset
    special_tokens_dict = {'additional_special_tokens': 
                            ['@USER', '<URL>']}  # hatexplain
    n_added_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    # print(tokenizer.all_special_tokens) 
    # print(tokenizer.all_special_ids)
    
    return tokenizer

# Label Encoding
le = LabelEncoder()
def encode(data):
    return le.fit_transform(data)

def decode(data):
    return le.inverse_transform(data)

def get_token_rationale(tokenizer, text, rationale, id):
    """
    # Example usage
    text = "good movie"              # Input words
    rationale = [1, 0]               # Rationale per word
    # If "good" gets tokenized to ["go", "##od"]
    # Output would be: [1, 1, 0]         # Rationale mapped to each token
    """
    text_token = tokenizer.tokenize(' '.join(text))
    assert len(text) == len(rationale), '[!] len(text) != len(rationale) | {} != {}\n{}\n{}'.format(len(text), len(rationale), text, rationale)
    
    rat_token = []
    for t, r in zip(text, rationale):
        token = tokenizer.tokenize(t)
        rat_token += [r]*len(token)

    assert len(text_token) == len(rat_token), "#token != #target rationales of {}".format(id)
    return rat_token

class GetLossAverage(object):
    """Compute average for torch.Tensor, used for loss average."""

    def __init__(self):
        self.reset()

    def add(self, v):
        count = v.data.numel()  # type -> int
        v = v.data.sum().item()  # type -> float
        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def aver(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res


def save_checkpoint(args, losses, model_state, trained_model):
    # checkpoint = {
    #     'args': args,
    #     'model_state': model_state,
    #     'optimizer_state': optimizer_state
    # }
    file_name = args.exp_name + '.ckpt'
    trained_model.save_pretrained(save_directory=os.path.join(args.dir_result, file_name))

    args.waiting += 1
    if losses[-1] <= min(losses):
        # print(losses)
        args.waiting = 0
        file_name = 'BEST_' + file_name
        trained_model.save_pretrained(save_directory=os.path.join(args.dir_result, file_name))
        
        if args.intermediate == 'mrp':
            # Save the embedding layer params
            emb_file_name = args.exp_name + '_emb.ckpt'
            torch.save(model_state.state_dict(), os.path.join(args.dir_result, emb_file_name))

        print("[!] The best checkpoint is updated")