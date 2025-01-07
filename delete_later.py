SEED = 42

from datasets import Dataset
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from src.utils.helpers import encode

from config import HF_TOKEN, WANDB_API_KEY


ds = load_dataset("sinhala-nlp/SOLD")
# prinst size of ds
print(ds)

sold_train = Dataset.to_pandas(load_dataset('sinhala-nlp/SOLD', split='train'))
sold_test = Dataset.to_pandas(load_dataset('sinhala-nlp/SOLD', split='test'))

# Display the first few rows and metadata about the dataset
print(sold_train.head())
print(sold_train.info())

# Display the first few rows and metadata about the dataset
print(sold_test.head())
print(sold_test.info())

# Print separateor
print("--------------------------------------------------")

trn_data = sold_train.rename(columns={'label': 'labels'})
tst_data = sold_test.rename(columns={'label': 'labels'})

# Display the first few rows and metadata about the dataset
print(trn_data.head())
print(trn_data.info())

# Display the first few rows and metadata about the dataset
print(tst_data.head())
print(tst_data.info())

print("--------------------------------------------------")
# load training data
train = trn_data[['text', 'labels']]
test = tst_data[['text', 'labels']]

# Print the first few rows of the training data
print(train.head())

# Print the first few rows of the test data
print(test.head())

# Print the shape of the training data
print(train.shape)

# Print the shape of the test data
print(test.shape)

print("--------------------------------------------------")

train['labels'] = encode(train["labels"])
test['labels'] = encode(test["labels"])

# Print the first few rows of the training data
print(train.head())

# Print the first few rows of the test data
print(test.head())

# Print the shape of the training data
print(train.shape)

# Print the shape of the test data
print(test.shape)

print("--------------------------------------------------")

test_sentences = test['text'].tolist()
train_df, eval_df = train_test_split(train, test_size=0.1, random_state=SEED)

# Print the shape of the training data
print(train_df.shape)

# Print the shape of the test data
print(eval_df.shape)