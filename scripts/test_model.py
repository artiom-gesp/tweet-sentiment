import pandas as pd
import string
import numpy as np
import evaluate
from transformers import TrainingArguments
from transformers import TrainingArguments, Trainer
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

device = 'cuda'

df_test = pd.read_table('test_data.txt', header=None, names=['tweet'])
df_test['stripped_tweets'] = df_test['tweet'].apply(strip_func)
strip_func = lambda x: x.lstrip(string.digits)[1:]


# Tokenizer is the same as for training
tokenizer = AutoTokenizer.from_pretrained("./TweetEval_roBERTa_5E")
# Training generates checkpoints, checkpoint name changes depending on hyperparameters, update path accordingly
test_model = AutoModelForSequenceClassification.from_pretrained('./test_trainer/checkpoint-24622', device_map='cuda')

correct = 0
predicted = []
for tw in df_test['stripped_tweets']:
  encoded_input = tokenizer(tw, return_tensors='pt').to('cuda')
  output = test_model(**encoded_input)
  scores = output[0][0].detach().cpu().numpy()
  if len(predicted) % 1000 == 0:
    print(len(predicted) / len(df_test['stripped_tweets']))
  predicted.append(1 if scores[1] > scores[0] else -1)

submission = pd.DataFrame(predicted, columns=['prediction'])
submission.index += 1
submission.to_csv('first_submission_lg_ha.csv', index_label='Id')
