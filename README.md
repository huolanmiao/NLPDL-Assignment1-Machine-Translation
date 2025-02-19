
In the upcoming assignments, you’ll be writing numerous Python and PyTorch programs. To get started with the coding basics, refer to these materials for preparation.

- Python review: https://web.stanford.edu/class/cs224n/readings/cs224n-python-review-2023.pdf
- Python Tutorial (Colab): https://colab.research.google.com/drive/1hxWtr98jXqRDs_rZLZcEmX_hUcpDLq6e?usp=sharing
- PyTorch Tutorial (Colab): https://colab.research.google.com/drive/1Pz8b_h-W9zIBk1p2e6v-YFYThG1NkYeS?usp=sharing

If you are already familiar with Python and PyTorch, then just skip these tutorials and finish the following tasks (but if you don’t, do read them carefully before asking TAs for help)!

## Task 1: Python Basics（20 Points）

Solving the two simple tasks. 

1. Flatten a list of lists. e.g., 

```python
Input = [[1,2,3], [4, 5], [6]]
Output = [1,2,3,4,5,6]
```

1. character count (only for lower case).

```python
Input = "hello world"
Output = {'h': 1, 'e': 1, 'l': 3, 'o': 2, 'w': 1, 'r': 1, 'd': 1}
```

[autograder.py](https://prod-files-secure.s3.us-west-2.amazonaws.com/0d5ed112-3783-475d-a67e-12ac9073a901/9fbc0713-36e1-4fda-a2c4-17a219bfb8fb/autograder.py)

[submission.py](https://prod-files-secure.s3.us-west-2.amazonaws.com/0d5ed112-3783-475d-a67e-12ac9073a901/80c05cbb-0499-43ce-8da3-2bac29b939ff/submission.py)

You can download the two python files and complete the functions in `submission.py`. Then you can run `autograder.py` to check if your code are correct (this is not an exhaustive check).

Then you need to design a large-scale experiment by yourself, i.e., increase the scale of the inputs. For example, you can prepare some very large lists that contain > 10^7 elements after flatting. Gradually increase the input scale (from 10^3, 10^4, 10^5, …, 10^7) and see how the time increase with your algorithm. Write a small report for this.

Can you come up with multiple ways of solving the above two problems (at least 2 different ways. e.g., using for-loop and using Python list comprehension are two different ways)? Also compare the efficiency (running time under large-scale input) in the report.

Note: To compute the running time, please do not read the running time from cells of Jupyter Notebook. It’s unreliable. The following operation is recommended:
```python
import time

begin = time.time()
... # your program here
end = time.time()

print("The runnining time = {}.".format(end - begin))
```

## Task 2: Text Classification with CNN（30 Points）

Read this [paper](https://arxiv.org/abs/1408.5882), and implement a CNN-based neural network for sentence classification (Chinese). The datasets are already processed as follows (each line is a datapoint with *text + label*), and you need to construct the vocabulary by yourself (you may need https://github.com/fxsjy/jieba to tokenize sentences and construct a vocabulary).

[dev.txt](https://prod-files-secure.s3.us-west-2.amazonaws.com/0d5ed112-3783-475d-a67e-12ac9073a901/c09a700e-ead8-4f89-b52c-4abe6a4e50dc/dev.txt)

[train.txt](https://prod-files-secure.s3.us-west-2.amazonaws.com/0d5ed112-3783-475d-a67e-12ac9073a901/f2c49495-64cb-4197-874a-711be8b3661e/train.txt)

[test.txt](https://prod-files-secure.s3.us-west-2.amazonaws.com/0d5ed112-3783-475d-a67e-12ac9073a901/1a0af9dc-6f53-45d7-9e72-fd6b3de8381d/test.txt)

Please make sure that:

- You cannot use any pre-trained model or pre-trained word vectors. The training should be from scratch.
- The accuracy should be higher than 75% on test set.
- You cannot use test set to tune the hyper-parameters. You should implement **early stopping** with the dev set.
    - If you are not familiar with early stopping (which is a popular regularization method in machine learning), just google it.
- You need to submit a tiny report that contains (1) configurations of your CNN model; (2) Classification accuracy on test set.

## Task 3: Machine Translation with RNN （50 Points）

In this part, you have to solve Japanese to English machine translation task with recurrent neural network. In your submission, you should show how to run your model clearly. (Wrapping in a .sh file is advised). You also need to include requirements.txt.

- Construct the vocabulary by yourself. You need to choose an appropriate way to tokenize Japanese and English. You need to randomly split the corpus into training set, validation set, and test set (the proportion = 8:1:1).
    
    [eng_jpn.txt](https://prod-files-secure.s3.us-west-2.amazonaws.com/0d5ed112-3783-475d-a67e-12ac9073a901/9a83f577-53ad-4320-9b06-affcd5555585/eng_jpn.txt)
    
- Train your word embedding using training set with any word embedding algorithm you like (e.g., CBOW, Skip-gram). Evaluate your trained word embedding with appropriate methods.
- Implement an LSTM with attention mechanism, and use the pre-trained word embedding as the embedding layer. Train your neural network on the training set.
- Evaluate your LSTM with BLEU score and Perplexity. Try to improve the BLEU score and Perplexity via some tricks (e.g., larger network…) we don’t require a very good BLEU but at least the model should output decent English words.
- You need to submit a tiny report that contains (1) configurations of your RNN model; (2) BLEU score and perplexity on training, validation, and test set (test set can not be seen during the entire training phase). (3) Show the prediction of your model on the following test case. (4) some analysis.
    
    ```python
    case_1 = "私の名前は愛です"
    case_2 = "昨日はお肉を食べません"
    case_3 = "いただきますよう"
    case_4 = "秋は好きです"
    case_5 = "おはようございます"
    ```
    

[NLP From Scratch: Translation with a Sequence to Sequence Network and Attention — PyTorch Tutorials 2.4.0+cu121 documentation](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)

**The code for all tasks should also be submitted.**