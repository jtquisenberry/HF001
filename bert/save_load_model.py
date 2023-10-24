# https://huggingface.co/docs/transformers/main_classes/model

from transformers import AutoModel

model = AutoModel.from_pretrained("bert-base-cased")

# Push the model to your namespace with the name "my-finetuned-bert".
model.push_to_hub("my-finetuned-bert")

# Push the model to an organization with the name "my-finetuned-bert".
model.push_to_hub("huggingface/my-finetuned-bert")