![telegram-cloud-photo-size-2-5418380851026191361-y](https://github.com/TimofeyZubashev/transformer_for_text_summarization/assets/120308225/f4dcacac-d0d2-4f49-b7a7-6dfbf0fc8cfd)

# Vanilla Transformer Architecture for Text Summarization

This repository features a text summarizer implemented using the vanilla transformer architecture.

## Who Might Find This Useful

1. **Learners**: Those who are learning about transformers and want to see a raw implementation of the transformer architecture.
2. **Experimenters**: Individuals looking to experiment with text summarization by modifying the existing model.

## How to Use This Repository

1. **Datasets**: The repository includes CNN news articles and their summaries (short versions). If you need longer versions, please download them from Kaggle.
2. **Training**: Run `train.py` to train the model. The model weights will be saved in the "folder2store_model_versions" directory.
3. **Testing**: Run `test_model_performance.py` to print the results in the terminal.

## How to Experiment

1. **Try Different Datasets**: Experiment with different datasets to see how the model performs.
2. **Modify the Loss Function**: The current implementation uses a combination of CrossEntropy and ROUGE loss (which measures the similarity of token embeddings). Experiment with different loss functions to improve performance.

Feel free to explore, modify, and enhance the model as you see fit!
