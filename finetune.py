import openai


def finetune_model(api_key, dataset_path):
    openai.api_key = api_key

    response = openai.FineTune.create(
        training_file=dataset_path,
        model="gpt-4o-2024-08-06",
        n_epochs=4,
        learning_rate_multiplier=0.1,
    )

    return response


# Example usage
# response = finetune_model("your_api_key_here", "path_to_your_dataset.jsonl")
