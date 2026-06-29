from datasets import Dataset, load_dataset
from transformers import AutoTokenizer

def load_pubmed_corpus(n=50000):
    
    dataset = Dataset.from_list(list(load_dataset("casinca/PUBMED_title_abstracts_2019_baseline", split="train", streaming=True).take(n)))
    return dataset


def tokenize_corpus(dataset, tokenizer):
    
    tokens = tokenizer(dataset["text"])
    return tokens


def group_into_blocks(tokenized, block_size=1024):
    flat_input = [item for sublist in tokenized['input_ids'] for item in sublist]
    flat_attention = [item for sublist in tokenized['attention_mask'] for item in sublist]

    n = len(flat_input) - (len(flat_input) % block_size)

    result = {
        "input_ids": [flat_input[i:i+block_size] for i in range(0, n, block_size)],
        "attention_mask": [flat_attention[i:i+block_size] for i in range(0, n, block_size)],
    }
    result["labels"] = result["input_ids"].copy()
    return result


"""
data = load_pubmed_corpus(10)
model_name = "Qwen/Qwen3-0.6B-Base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

tokenized_data = data.map(lambda x: tokenize_corpus(x, tokenizer), batched= True, remove_columns= data.column_names)

print(tokenized_data)
print(tokenized_data[0])
print(len(tokenized_data[0]["input_ids"]))

blocks = tokenized_data.map(lambda x: group_into_blocks(x, block_size=128), batched= True)
print(blocks)

print(blocks[0])
"""