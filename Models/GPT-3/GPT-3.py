import openai
import json
import os

# Step 1: Convert the dataset to fine-tuning format (JSONL)
def convert_to_finetune_format(json_file, output_file):
    """ Convert the dataset to OpenAI GPT-3 fine-tuning JSONL format. """
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        with open(output_file, 'w', encoding='utf-8') as out_file:
            for item in data:
                prompt = item['question'] + "\nAnswer:"
                completion = " " + item['answer']
                out_file.write(json.dumps({"prompt": prompt, "completion": completion}) + "\n")
        
        print(f"Dataset successfully converted to {output_file}")
    except FileNotFoundError:
        print(f"Error: {json_file} not found.")
    except json.JSONDecodeError:
        print(f"Error: {json_file} is not properly formatted as JSON.")

# Step 2: Fine-tune GPT-3 model using OpenAI CLI
def fine_tune_gpt3_via_cli(jsonl_file, model="davinci"):
    """ Fine-tune GPT-3 using OpenAI CLI. """
    try:
        # Upload dataset and fine-tune via OpenAI CLI
        os.system(f"openai tools fine_tunes.prepare_data -f {jsonl_file}")
        os.system(f"openai api fine_tunes.create -t {jsonl_file} -m {model}")
        print(f"Fine-tuning started for {jsonl_file}")

    except Exception as e:
        print(f"An error occurred during fine-tuning: {e}")

# Step 3: Use the fine-tuned model to predict answers
def predict_gpt3(api_key, prompt, model_name="gpt-3.5-turbo", stop=["\n"], max_tokens=100, temperature=0.5):
    """ Generate prediction using fine-tuned GPT-3 model. """
    openai.api_key = api_key
    try:
        response = openai.Completion.create(
            model=model_name,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop
        )
        return response['choices'][0]['text'].strip()
    except openai.error.OpenAIError as e:
        print(f"OpenAI API error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Step 4: Evaluate the model's performance
def evaluate_performance(api_key, dataset_file, model_name="gpt-3.5-turbo"):
    """ Evaluate GPT-3 model on accuracy, F1 score and other benchmarks. """
    with open(dataset_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    y_true = []
    y_pred = []

    for item in data:
        prompt = item['question'] + " Answer:"
        true_answer = item['answer']
        predicted_answer = predict_gpt3(api_key, prompt, model_name=model_name)
        
        y_true.append(true_answer.strip().lower())
        y_pred.append(predicted_answer.strip().lower())

    # Calculate Accuracy and F1 score
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    return accuracy, f1

# Step 5: Process multiple datasets (UMLC, MATH, GSM8K)
def process_datasets(dataset_dir, datasets, api_key, model_name="gpt-3.5-turbo"):
    """ Process multiple datasets and fine-tune GPT-3 for each one. """
    performance_results = {}
    
    for dataset_name in datasets:
        print(f"Processing dataset: {dataset_name}")
        json_file = os.path.join(dataset_dir, f"{dataset_name}.json")
        jsonl_file = os.path.join(dataset_dir, f"{dataset_name}_gpt3_finetune.jsonl")

        # Convert dataset to fine-tune format
        convert_to_finetune_format(json_file, jsonl_file)

        # Fine-tune GPT-3 using OpenAI CLI
        fine_tune_gpt3_via_cli(jsonl_file, model=model_name)

        # Evaluate the model's performance
        accuracy, f1 = evaluate_performance(api_key, json_file, model_name=model_name)

        performance_results[dataset_name] = {
            "accuracy": accuracy,
            "f1_score": f1
        }
        print(f"Performance for {dataset_name}: Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")

    return performance_results

# Step 6: Example of prediction using the fine-tuned model
def main():
    api_key = 'your-openai-api-key'

    # Directory containing the datasets
    dataset_dir = r'C:\Users\micha\UMLC_Project\datasets'
    
    # List of datasets to process
    datasets = ['UMLC', 'MATH', 'GSM8K']

    # Process each dataset and evaluate performance
    performance_results = process_datasets(dataset_dir, datasets, api_key)

    # Output performance results
    print("\nFinal Performance Results:")
    for dataset, results in performance_results.items():
        print(f"{dataset} - Accuracy: {results['accuracy']:.4f}, F1 Score: {results['f1_score']:.4f}")

if __name__ == "__main__":
    main()
