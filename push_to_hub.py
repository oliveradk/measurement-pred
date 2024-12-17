from models import load_model, get_model_id, ModelTypes

model_paths = [
    # (0, "multirun/2024-12-16/08-54-40/0/checkpoint-3905"), 
    # (1, "multirun/2024-12-16/09-38-37/0/checkpoint-3905"), 
    # (2, "multirun/2024-12-16/09-40-07/0/checkpoint-3905"), 
    # (3, "multirun/2024-12-16/09-44-11/0/checkpoint-3905"), 
    # (5, "multirun/2024-12-16/09-46-44/0/checkpoint-3905"), 
    # (6, "multirun/2024-12-16/09-46-45/0/checkpoint-3905"), 
]

model_type = ModelTypes.AUTO_MODEL_FOR_SEQUENCE_CLASSIFICATION

for i, model_path in model_paths:

    model_id = get_model_id("Salesforce/codegen-350M-mono", f"redwoodresearch/diamonds-seed{i}")

    config, model, tokenizer = load_model(model_type, model_path)

    model.push_to_hub(model_id)