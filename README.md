# SentenceTransformerModel

This repository contains the implementation of a PyTorch-based sentence transformer model, which provides task-specific heads for both classification and sentiment analysis.

## Overview

The `SentenceTransformerModel` leverages a pre-trained transformer model (e.g., BERT) for sentence embeddings and includes two task-specific heads:

- **Classification Head**: For general-purpose sentence classification tasks with a configurable number of output classes.
- **Sentiment Analysis Head**: Specifically designed to classify sentences into three sentiment categories: `positive`, `neutral`, and `negative`.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/zhaozb08/sentence-transformer
   cd sentence-transformer
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Tasks

### Task 1:

1. Command:
    ```bash
    python3 task_1.py
    ```
    Model class: `SentenceTransformerModel` (models.py)

2. Output:

    Input sentences:

    ['This repository is useful to generate sentence embeddings', 'Similar sentences have similar embeddings', 'I have a little cat', 'I have a kitten']

    Embeddings:

    tensor([[-0.0960, -0.2277,  0.0198,  ..., -0.0443, -0.3546,  0.1897],
            [ 0.0181, -0.0697, -0.2416,  ...,  0.0345, -0.1594,  0.4157],
            [-0.0563, -0.1428,  0.2112,  ..., -0.1178,  0.4658,  0.0745],
            [-0.0983, -0.0544,  0.1672,  ..., -0.0341,  0.3767,  0.0280]])

    The similarity score between 'I have a little cat' and 'I have a kitten' is 0.93

    The similarity score between 'This repository is useful to generate sentence embeddings' and 'I have a little cat' is 0.42

3. Choices:

- **Pre-trained Model**: I'm using `bert-base-uncased` by default because it's generic and widely adopted. We can consider other pretrained checkpoints (e.g., `distilbert-base-uncased`, `roberta-base`) for specific requirements.
- **Mean Pooling**: The implementation uses mean pooling on the last hidden states to create fixed-size embeddings. Other options include CLS token embedding, max pooling, and weighted pooling. Mean pooling is more robust in general since each token contributes equally. We can try other methods in practice and compare.
- **Embedding Dimensions**: Using 768 for `bert-base-uncased`. We can project it to lower dimensions when there are requirements on it. 
Device Compatibility: The code detects GPU availability and moves the tensors to the appropriate device.

### Task 2

#### Classification

1. Command:
    ```bash
    python3 task_2a.py
    ```
    Model class: `SentenceClassifier` (models.py)

    The classification head can be used for any general-purpose sentence classification task. Set the `num_classes` parameter in the model to match the number of output classes.

2. Sample Output (since the head randomly initialized): 

    Input sentences: ['This is a spam message.', 'This is not a spam message.']

    Logits: tensor([[-0.0705,  0.1392],
            [-0.0511,  0.0884]])

    Probabilities: tensor([[0.4478, 0.5522],
            [0.4652, 0.5348]])

    Predicted Classes: ['not spam', 'not spam']

#### Multi-Task With Sentiment Analysis

1. Command:
    ```bash
    python3 task_2b.py
    ```
    Model class: `MultiHeadSentenceTransformer` (models.py)

2. Sample Output (since the head is randomly initialized):

    Input sentences:

    ['I love this product!', 'The product is okay, not great.', 'The product is defective.']

    Sentiment Logits: tensor([[-0.0428, -0.2051,  0.0146],
            [-0.0044, -0.2357, -0.0293],
            [-0.0428, -0.2430, -0.0300]])

    Sentiment Probabilities: tensor([[0.3437, 0.2922, 0.3641],
            [0.3612, 0.2866, 0.3523],
            [0.3532, 0.2891, 0.3577]])

    Sentiment Classes: ['negative', 'positive', 'negative']


### Discussions

1. Which Parameters to Freeze

- **Freeze the Transformer Backbone:** When working with small datasets or closely related tasks to the pre-training data.
- **Train the Entire Network:** When datasets are large, tasks are domain-specific, or differ significantly from pre-training tasks.
- **Freeze One Head:** When tasks are independent or datasets are imbalanced.
- **Train Both Heads:** For related tasks with balanced datasets, leveraging multi-task learning.

2. Multi-Task Model VS Separate Models

    Key considerations:
- **Task Similarity:** Use a multi-task model if tasks are related; otherwise, use separate models.
- **Dataset Size:** Use multi-task learning if one task has limited data; otherwise, use separate models if both have sufficient data.
- **Computational Resources:** Use a multi-task model if inference efficiency is critical.
- **Modularity Needs:** Use separate models if tasks require independent updates or specialized architectures.

3. Handling Data Imbalance

    Assuming Task 1 has sufficient data while Task 2 has limited data:

- **Sampling:** Down sample Task 1 data and/or up sample Task 2 data; generate synthetic data for Task 2 if possible
- **Weighted Loss Functions:** Use a higher weight on Task 2's loss function
- **Gradient/Learning Rate Scaling:** Similar to weighted loss function
- **Regularization for Task 2:** To avoid overfitting
- **Freeze Task 1 Head:** Run more training iterations for Task 2

## Dependencies

- `torch`
- `transformers`
- `numpy`

## License

This project is licensed under the MIT License. See the LICENSE file for details.