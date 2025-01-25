import torch
from models import SentenceTransformerModel, SentenceClassifier

if __name__ == "__main__":
    sentences = [
        "This is a spam message.",
        "This is not a spam message."
    ]
    print("Input sentences:", sentences)
    classes = ['spam', 'not spam']

    base_model = SentenceTransformerModel("bert-base-uncased")
    sentence_classifier = SentenceClassifier(base_model, num_classes=2)

    with torch.no_grad():
        logits = sentence_classifier(sentences)
        probabilities = torch.softmax(logits, dim=1)
        predicted_classes = torch.argmax(probabilities, dim=1)

    print("Logits:", logits)
    print("Probabilities:", probabilities)
    print("Predicted Classes:", [classes[i] for i in predicted_classes])