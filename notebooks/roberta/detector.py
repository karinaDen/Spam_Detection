import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split
from transformers import RobertaTokenizer, RobertaForSequenceClassification, get_linear_schedule_with_warmup
from roberta.utils.metrics import compute_metrics, confusion_matrix
from roberta.utils.plotting import plot_heatmap, plot_roc_auc
from roberta.utils.seed import random_seed
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelBinarizer


import matplotlib.pyplot as plt

from tqdm import tqdm

def save_list_to_file(lst, filename):
    with open(filename, 'w') as file:
        for item in lst:
            file.write(str(item) + '\n')

class SpamMessageDetector:
    def __init__(self, model_path, max_length=512, seed=0):
        random_seed(seed)
        self.seed = seed
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = RobertaTokenizer.from_pretrained(model_path)
        self.model = RobertaForSequenceClassification.from_pretrained(model_path, num_labels=2)
        self.model = self.model.to(self.device)
        self.max_length = max_length
    

    def evaluate(self, dataset_path):
        random_seed(self.seed)

        # Load and preprocess the dataset
        dataset = pd.read_csv(dataset_path)
        texts = dataset["text"].tolist()
        labels = dataset["text_type"].tolist()
        
        def preprocess(text):
            inputs = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=self.max_length,
                padding="longest",
                truncation=True,
                return_tensors="pt"
            )
            return inputs["input_ids"].to(self.device), inputs["attention_mask"].to(self.device)

        inputs = [preprocess(text) for text in texts]

        # Make predictions on the dataset
        predictions = []
        probabilities = []

        with torch.no_grad():
            for input_ids, attention_mask in inputs:
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                predicted_label = torch.argmax(logits, dim=1).item()
                if predicted_label == 0:
                    predictions.append("ham")
                else:
                    predictions.append("spam")

                # Calculate the probability of the positive class
                proba = torch.nn.functional.softmax(logits, dim=1)[:, 1].item()
                probabilities.append(proba)
    
                    
        # compute evaluation metrics
        accuracy, precision, recall, f1, roc_auc = compute_metrics(labels, predictions)

        # Create confusion matrix
        cm = confusion_matrix(labels, predictions)
        labels_sorted = sorted(set(labels))

        # Print evaluation metrics
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}")

        # Convert labels to binary
        lb = LabelBinarizer()
        y_true = lb.fit_transform(labels).flatten()  # "spam" becomes 1, "ham" becomes 0

        # Compute ROC curve
        fpr, tpr, _ = roc_curve(y_true, probabilities)


        # Plot the confusion matrix
        plot_heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels_sorted, yticklabels=labels_sorted)

        # Plot the ROC curve
        plot_roc_auc(fpr, tpr)
        
    
    def detect(self, text):
        random_seed(self.seed)
        is_str = True
        if isinstance(text, str):
            encoded_input = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
        elif isinstance(text, list):
            is_str = False
            encoded_input = self.tokenizer.batch_encode_plus(
                text,
                add_special_tokens=True,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
        else:
            raise Exception("text type is unsupported, needs to be str or list(str)")

        input_ids = encoded_input['input_ids'].to(self.device)
        attention_mask = encoded_input['attention_mask'].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)

        logits = outputs.logits
        predicted_labels = torch.argmax(logits, dim=1).tolist()
        proba = torch.nn.functional.softmax(logits, dim=1)[:, 1].tolist()
        
        if is_str: 
            return predicted_labels[0], proba[0]
        else:
            return predicted_labels, proba
    
    def save_model(self, model_path):
        self.model.save_pretrained(model_path)
        self.tokenizer.save_pretrained(model_path)

    def load_model(self, model_path):
        self.model = RobertaForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = RobertaTokenizer.from_pretrained(model_path)
        self.model = self.model.to(self.device)