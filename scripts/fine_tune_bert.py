import pandas as pd
import string
import torch
import numpy as np
import evaluate
from transformers import TrainingArguments
from transformers import TrainingArguments, Trainer
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Running the code on Euler:
"""
- ssh to cluser
- clone repo
- RUN: sbatch --mem-per-cpu=16G --gpus=1 --gres=gpumem:24g ./fine_tune_bert.sh
- Check stdout: cat slurm-XXX.out
- Check job status: squeue --all
"""



DEBUG = False
device = 'cuda'
output_dir = '/cluster/home/argesp/cil/test_trainer'
# install -> restart

"""### Data Loading"""

df_train_pos = pd.read_table('train_pos.txt', header=None, names=['tweet'])
df_train_neg = pd.read_table('train_neg.txt', header=None, names=['tweet'])

df_train_pos['sentiment'] = 1
df_train_neg['sentiment'] = 0

df_train = pd.concat([df_train_pos, df_train_neg], names=['tweet'])


# Loading the model to fine tune (could not make euler work without having it installed locally)
tokenizer = AutoTokenizer.from_pretrained("./TweetEval_roBERTa_5E")

model = AutoModelForSequenceClassification.from_pretrained("./TweetEval_roBERTa_5E", num_labels=2, device_map=device)


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


__train_dataset = df_train
_train_dataset = __train_dataset.sample(frac=1, random_state=1)
train_dataset = Dataset.from_pandas(_train_dataset)
train_dataset = train_dataset.rename_column("tweet", "text").rename_column('sentiment', 'label')
train_datasets = train_dataset.map(tokenize_function, batched=True)

test_dataset = Dataset.from_pandas(__train_dataset.drop(_train_dataset.index))
test_dataset = test_dataset.rename_column("tweet", "text").rename_column('sentiment', 'label')
test_datasets = test_dataset.map(tokenize_function, batched=True)


metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


training_args = TrainingArguments(
   output_dir=output_dir,
   learning_rate=2e-5,
   per_device_train_batch_size=16,
   per_device_eval_batch_size=16,
   num_train_epochs=2,
   weight_decay=0.01,
   save_strategy="epoch",
)

trainer = Trainer(
   model=model,
   args=training_args,
   train_dataset=train_datasets,
   eval_dataset=test_datasets,
   tokenizer=tokenizer,
   compute_metrics=compute_metrics,
)

trainer.train()
