SEED = 42

from datasets import Dataset
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from src.utils.helpers import encode

sold_train = Dataset.to_pandas(load_dataset('sinhala-nlp/SOLD', split='train'))
sold_test = Dataset.to_pandas(load_dataset('sinhala-nlp/SOLD', split='test'))

trn_data = sold_train.rename(columns={'label': 'labels'})
tst_data = sold_test.rename(columns={'label': 'labels'})


# load training data
train = trn_data[['text', 'labels']]
test = tst_data[['text', 'labels']]

train['labels'] = encode(train["labels"])
test['labels'] = encode(test["labels"])

test_sentences = test['text'].tolist()
train_df, eval_df = train_test_split(train, test_size=0.1, random_state=SEED)