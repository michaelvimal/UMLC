import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW
from sklearn.metrics import accuracy_score, f1_score
import json
import os

# Step 1: Load your dataset of math word problems
class MathWordProblemDataset(Dataset):
    def __init__(self, data_file, tokenizer, max_length=512):
        # Specify the 'utf-8' encoding while opening the file to avoid UnicodeDecodeError
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question = self.data[idx]['question']
        answer = self.data[idx]['answer']
        
        # Concatenate question and answer for input-output pairs
        input_text = question + " Answer:"
        output_text = answer

        # Tokenize input and output
        input_encoding = self.tokenizer(
            input_text, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt"
        )
        target_encoding = self.tokenizer(
            output_text, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt"
        )

        input_ids = input_encoding['input_ids'].squeeze().to(device)  # Move to GPU
        attention_mask = input_encoding['attention_mask'].squeeze().to(device)  # Move to GPU
        labels = target_encoding['input_ids'].squeeze().to(device)  # Move to GPU

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

# Step 2: Split dataset into train, validation, and test sets
def split_dataset(dataset, train_split=0.7, val_split=0.15):
    train_size = int(train_split * len(dataset))
    val_size = int(val_split * len(dataset))
    test_size = len(dataset) - train_size - val_size

    return random_split(dataset, [train_size, val_size, test_size])

# Step 3: Prepare the DataLoader for train/validation/test
def prepare_data(json_file, tokenizer, batch_size=16):
    dataset = MathWordProblemDataset(json_file, tokenizer)
    
    train_data, val_data, test_data = split_dataset(dataset)
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader

# Step 4: Define the GPT-2 Model
def build_model(device):
    # Load the GPT-2 model with language modeling head
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model = model.to(device)  # Move model to GPU
    return model

# Step 5: Train the GPT-2 Model
def train_model(model, dataloader, optimizer, num_epochs=5):
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(dataloader)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')

# Step 6: Evaluate Model Performance
def evaluate_model(model, dataloader):
    model.eval()
    predictions, true_labels = [], []
    total_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])
            loss = outputs.loss
            total_loss += loss.item()

            generated_ids = model.generate(batch['input_ids'], max_length=100, num_beams=5, early_stopping=True)
            predictions.extend(generated_ids.cpu().numpy())  # Move predictions back to CPU
            true_labels.extend(batch['labels'].cpu().numpy())  # Move labels back to CPU

    avg_loss = total_loss / len(dataloader)
    f1 = f1_score(true_labels, predictions, average='weighted')
    accuracy = accuracy_score(true_labels, predictions)
    
    return avg_loss, accuracy, f1

# Step 7: Predict Using the GPT-2 Model
def predict_model(model, tokenizer, question):
    model.eval()

    # Tokenize the input question
    input_text = question + " Answer:"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding='max_length', max_length=512).to(device)

    # Generate output
    with torch.no_grad():
        generated_ids = model.generate(inputs.input_ids, max_length=100, num_beams=5, early_stopping=True)
    
    # Decode the output tokens to get the predicted answer
    prediction = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return prediction

# Step 8: Load MATH and GSM8K datasets
def load_datasets(dataset_dir, tokenizer, batch_size=16):
    datasets = ['UMLC', 'MATH', 'GSM8K']
    dataloaders = {}
    
    for dataset_name in datasets:
        json_file = os.path.join(dataset_dir, dataset_name, f"{dataset_name}.json")
        train_loader, val_loader, test_loader = prepare_data(json_file, tokenizer, batch_size)
        dataloaders[dataset_name] = (train_loader, val_loader, test_loader)
    
    return dataloaders

def main():
    dataset_dir = r'C:\Users\micha\UMLC Project\datasets'

    # Move model to GPU if available
    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize the tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token  # Set the padding token to be the EOS token
    model = build_model(device)  # Pass device to build_model

    # Load datasets for UMLC, MATH, and GSM8K
    dataloaders = load_datasets(dataset_dir, tokenizer)

    # Train and evaluate on each dataset
    for dataset_name, (train_loader, val_loader, test_loader) in dataloaders.items():
        print(f"Training on {dataset_name} dataset...")
        optimizer = AdamW(model.parameters(), lr=5e-5)

        # Train the model
        train_model(model, train_loader, optimizer, num_epochs=3)

        # Evaluate the model
        val_loss, val_accuracy, val_f1 = evaluate_model(model, val_loader)
        print(f"Validation Performance for {dataset_name} - Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}, F1 Score: {val_f1:.4f}")

        # Test the model
        test_loss, test_accuracy, test_f1 = evaluate_model(model, test_loader)
        print(f"Test Performance for {dataset_name} - Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}, F1 Score: {test_f1:.4f}")

    # Predict using a sample question
    sample_question = "John has 5 apples, he gives 2 apples away. How many apples does he have?"
    prediction = predict_model(model, tokenizer, sample_question)
    print(f"Prediction: {prediction}")

if __name__ == "__main__":
    main()
