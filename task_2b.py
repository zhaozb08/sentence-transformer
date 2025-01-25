import torch
from models import SentenceTransformerModel, MultiHeadSentenceTransformer

if __name__ == "__main__":
    sentences = [
        "I love this product!",  # positive
        "The product is okay, not great.",  # neutral
        "The product is defective."  # negative
    ]
    print("Input sentences:")
    print(sentences)

    classes = ["positive", "neutral", "negative"]

    # Instantiate the model
    base_model = SentenceTransformerModel("bert-base-uncased")
    multi_head_classifier = MultiHeadSentenceTransformer(base_model, num_classes_1=2, num_classes_2=3)
    # Forward pass for classification task
    with torch.no_grad():
        classifier_logits, sentiment_logits = multi_head_classifier(sentences)
    sentiment_probabilities = torch.softmax(sentiment_logits, dim=1)
    sentiment_classes = torch.argmax(sentiment_probabilities, dim=1)

    print("Sentiment Logits:", sentiment_logits)
    print("Sentiment Probabilities:", sentiment_probabilities)
    print("Sentiment Classes:", [classes[i] for i in sentiment_classes])