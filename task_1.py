import torch
from models import SentenceTransformerModel
import numpy as np

# a similarity function to characterize the similarity between sentences
def cos_sim(vec_1, vec_2):
    return np.dot(vec_1, vec_2) / (np.linalg.norm(vec_1) * np.linalg.norm(vec_2))

if __name__ == "__main__":
    
    sentences = [
        "This repository is useful to generate sentence embeddings",
        "Similar sentences have similar embeddings",
        "I have a little cat",
        "I have a kitten",
    ]

    print("Input sentences:")
    print(sentences)

    model = SentenceTransformerModel("bert-base-uncased")
    with torch.no_grad():
        embeddings = model(sentences)
    
    print("Embeddings:")
    print(embeddings)

    # Calculate some similarity scores for fun!
    sim_1 = cos_sim(embeddings[2], embeddings[3])
    sim_2 = cos_sim(embeddings[0], embeddings[2])
    print(f"The similarity score between '{sentences[2]}' and '{sentences[3]}' is {sim_1:.2f}")
    print(f"The similarity score between '{sentences[0]}' and '{sentences[2]}' is {sim_2:.2f}")
