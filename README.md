Linear and Quadratic Discriminant Analysis
==========================================

Implemented these two Machine Learning algorithms while studying CS 189 @ UC Berkeley.

Project Outline:

- `classifier.py`  - machine learning code of LDA and QDA
- `train.py`       - sample training code using LDA or QDA with MNIST or SPAM
- `data/spam.mat`  - matlab file containing spam training data
  + each row represents an email labeled as either "spam" or "ham"
  + each col represents frequency of a word (actual data is l1-normalized)
- `data/spam.obj`  - python pickle file containing dictionary of words
- `data/mnist.mat` - matlab file containing mnist digit training data
  + each row represents a 28-by-28 flatten greyscale image
  + each col represents greyscale from 0 to 255 (actual data is l2-normalized)

The data files are not included in this github repo, they can be downloaded [here](https://www.dropbox.com/sh/6w5do8nvydov549/AAAcgs4vUwY7vnmrNn9EfeqZa?dl=0).
