import torch
from mwptoolkit.data.dataset import Dataset
from mwptoolkit.data.dataloader import DataLoader
from mwptoolkit.model.Seq2Tree.graph2tree import Graph2Tree
from mwptoolkit.utils.preprocess_tool import PreprocessTool
from mwptoolkit.utils.loss import MaskedCrossEntropyLoss
from mwptoolkit.trainer import Trainer
from sklearn.metrics import accuracy_score, f1_score
import json
import os

# Step 1: Load your math word problem dataset
def load_math_word_problem_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

# Step 2: Prepare Dataset and DataLoader
def prepare_data(json_path, batch_size=32):
    # Load the dataset
    dataset = Dataset(dataset_path=json_path)
    
    # Preprocess the data
    preprocess_tool = PreprocessTool()
    dataset = preprocess_tool.preprocess_dataset(dataset)
    
    # Split dataset into training and testing sets
    train_data, test_data = dataset.split_dataset(train_split=0.8, test_split=0.2)

    # DataLoader for batching and shuffling data
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, dataset

# Step 3: Define the Graph2Tree Model
def build_model(vocab_size, output_lang_size, device):
    # Create a Graph2Tree model
    model = Graph2Tree(
        embedding_size=128,
        hidden_size=256,
        vocab_size=vocab_size,
        output_lang_size=output_lang_size,
        dropout=0.5,
        device=device
    )
    return model

# Step 4: Train the model
def train_model(model, train_loader, vocab_size, output_lang_size, device):
    # Define optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_function = MaskedCrossEntropyLoss()
    
    # Initialize the trainer
    trainer = Trainer(
        model=model,
        train_data=train_loader,
        dev_data=None,  # Optionally, you can add a validation set
        optimizer=optimizer,
        criterion=loss_function,
        vocab_size=vocab_size,
        output_lang_size=output_lang_size,
        epochs=10,
        device=device
    )
    
    # Train the model
    trainer.train()

# Step 5: Evaluate Model Performance
def evaluate_model(model, test_loader, device):
    model.eval()
    predictions, true_labels = [], []
    total_loss = 0

    loss_function = MaskedCrossEntropyLoss()
    
    with torch.no_grad():
        for batch in test_loader:
            input_seq = batch['input_seq'].to(device)
            target_seq = batch['target_seq'].to(device)
            
            # Forward pass
            outputs = model(input_seq)
            loss = loss_function(outputs, target_seq)
            total_loss += loss.item()

            # Predict and store
            prediction = model.predict(input_seq)
            predictions.extend(prediction.cpu().numpy())  # Move predictions back to CPU
            true_labels.extend(target_seq.cpu().numpy())  # Move labels back to CPU

    avg_loss = total_loss / len(test_loader)
    
    # Calculate Accuracy and F1 Score
    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average='weighted')
    
    return avg_loss, accuracy, f1

# Step 6: Process all datasets (UMLC, MATH, GSM8K) and calculate performance
def process_datasets(dataset_dir, datasets, batch_size=32):
    # Set device to GPU if available, else use CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Iterate over each dataset
    for dataset_name in datasets:
        print(f"Processing {dataset_name} dataset...")
        
        # Prepare dataset paths
        json_file_path = os.path.join(dataset_dir, dataset_name, f"{dataset_name}.json")

        # Load and prepare data
        train_loader, test_loader, dataset = prepare_data(json_file_path, batch_size)
        
        # Get the vocabulary size for input and output
        vocab_size = len(dataset.vocab['input_vocab'])
        output_lang_size = len(dataset.vocab['output_vocab'])
        
        # Build the Graph2Tree model and move it to the appropriate device (GPU/CPU)
        model = build_model(vocab_size, output_lang_size, device).to(device)
        
        # Train the model
        train_model(model, train_loader, vocab_size, output_lang_size, device)
        
        # Evaluate the model on the test set and calculate performance metrics
        avg_loss, accuracy, f1 = evaluate_model(model, test_loader, device)
        
        # Print out the performance metrics
        print(f"Performance for {dataset_name} dataset:")
        print(f"Test Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")

def main():
    dataset_dir = r'C:\Users\micha\UMLC Project\datasets'
    
    # List of datasets to process
    datasets = ['UMLC', 'MATH', 'GSM8K']
    
    # Process all datasets
    process_datasets(dataset_dir, datasets)

if __name__ == "__main__":
    main()
