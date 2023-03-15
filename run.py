from datasets import load_dataset

dataset = load_dataset("OxAISH-AL-LLM/wiki_toxic")
print(dataset['train'][100])

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("law-ai/InLegalBERT")

def tokenize_func(ex):
    return tokenizer(ex['comment_text'], padding='max_length', truncation=True)

tokenized_datasets = dataset.map(tokenize_func, batched=True)

small_train_dataset = tokenized_datasets['train'].shuffle(seed=42).select(range(5000))

from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained('law-ai/InLegalBERT')

from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(output_dir='inlegalbert_trainer', use_mps_device=True)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
)

trainer.train()