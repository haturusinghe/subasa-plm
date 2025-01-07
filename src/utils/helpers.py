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
    special_tokens_dict = {'additional_special_tokens': 
                            ['<user>', '<number>']}  # hatexplain
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
