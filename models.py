import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

# base sentence embedding model for Task 1
class SentenceTransformerModel(nn.Module):
    def __init__(self, model_name: str = "bert-base-uncased"):
        super(SentenceTransformerModel, self).__init__()

        self.encoder = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Mean pooling layer for sentence embeddings
        self.pooling = nn.AdaptiveAvgPool1d(1)

    def forward(self, input_texts):
        # Tokenize input sentences
        encoded_input = self.tokenizer(
            input_texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )

        # Move tensors to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = self.encoder.to(device)
        encoded_input = {key: val.to(device) for key, val in encoded_input.items()}

        # Get hidden states from the transformer
        output = self.encoder(**encoded_input, return_dict=True)

        # Use the mean of the last hidden state for sentence embeddings
        last_hidden_state = output.last_hidden_state
        sentence_embeddings = last_hidden_state.mean(dim=1)

        return sentence_embeddings

# classification model for Task 2A
class SentenceClassifier(nn.Module):
    def __init__(self, base_model: SentenceTransformerModel, num_classes: int):
        super(SentenceClassifier, self).__init__()
        self.base_model = base_model
        
        # New classification head for additional tasks
        hidden_size = self.base_model.encoder.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, num_classes)
        )

    def forward(self, input_texts):
        # Get sentence embeddings from the base model
        sentence_embeddings = self.base_model(input_texts)
        # Pass sentence embeddings through the new classifier
        logits = self.classifier(sentence_embeddings)
        return logits

# Multi-head model for Task 2B
class MultiHeadSentenceTransformer(nn.Module):
    def __init__(self, base_model: SentenceTransformerModel, num_classes_1: int = 2, num_classes_2: int = 3):
        super(MultiHeadSentenceTransformer, self).__init__()
        self.base_model = base_model
        hidden_size = self.base_model.encoder.config.hidden_size
        
        self.new_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, num_classes_1)
        )

        # Classification head
        self.classifier_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, num_classes_1)
        )

        # Sentiment head
        self.sentiment_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 3)  # 3 classes: positive, neutral, negative
        )

    def forward(self, input_texts, task: str = "classification"):
        
        sentence_embeddings = self.base_model(input_texts)
        # classifier
        classifier_logits = self.classifier_head(sentence_embeddings)
        sentiment_logits = self.sentiment_head(sentence_embeddings)

        return classifier_logits, sentiment_logits


