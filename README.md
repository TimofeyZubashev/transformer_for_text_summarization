![telegram-cloud-photo-size-2-5418380851026191361-y](https://github.com/TimofeyZubashev/transformer_for_text_summarization/assets/120308225/f4dcacac-d0d2-4f49-b7a7-6dfbf0fc8cfd)

# Vanila Transformer Architecture for text summarization

In this folder I have realised transformer summarizer using vanila transformer achitecture.

This repository can be helpful for thouse who:

1) Learning transformers and wants so see raw transformer architecture realsiation

2) Wants to experiment with text summariztion by adding editing existing model

# How to use rep

1) Datasets include CNN news and their summaries, but short version, download on Kaggle if needed longer

2) Run train.py -> train model and store model weights in "folder2store_model_versions"

3) Run test_model_performance.py to see results printed in terminal

# How to experiment

1) I suggest you try different datasets

2) I suggest you experiment with loss function (I am using combination of CrossEntropy + ROGUE loss (which measues vector simmilarities of alike token embedding))
