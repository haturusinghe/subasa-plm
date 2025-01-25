import os
from datasets import load_dataset, Dataset

hf_sold_train = load_dataset('sinhala-nlp/SOLD', split='train')
sold_train = Dataset.to_pandas(hf_sold_train)

hf_sold_test = load_dataset('sinhala-nlp/SOLD', split='test')
sold_test = Dataset.to_pandas(hf_sold_test)

# check if an empty directory called SOLD_DATASET exists in the current working directory
if not os.path.exists('SOLD_DATASET'):
    os.makedirs('SOLD_DATASET')

sold_train.to_json(
    'SOLD_DATASET/sold_train_split.json',
    orient='records',
    indent=4,
    force_ascii=False,
    date_format='iso'
)

sold_test.to_json(
    'SOLD_DATASET/sold_test_split.json',
    orient='records',
    indent=4,
    force_ascii=False,
    date_format='iso'
)