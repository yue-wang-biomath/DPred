# DPred
## 1. Description
This is a novel computational tool for predicting Dihydrouridine (D) sites over mRNA sequences. This modification can serve as a metabolic modulator for various pathological conditions, and its elevated levels in tumors are associated with a series of cancers. Precise identification of D sites on RNA is vital for understanding its biological function. DPred was maily built upon the additive local self-attention and convolutional neural network architecture. 

## 2. Requirements
Before prediction, please make sure the following packages are installed in the Python environment:
Python = 3.10.0
Tensorflow = 2.10.0
Numpy 1.22.3
matplotlib = 3.5.3
pandas = 1.4.4 
scikit-learn = 1.1.1
keras_self_attention = 0.51.0
keras = 2.10.0

## 3.Results
Please prepare a Fasta file including 41 nt sequences. In the center of a sequence should be a potential D site to be evaluated.  
The prediction results are summarized in the file results file, including four columns of 'Sequence ID', 'Sequence', 'Probability' and 'Prediction Result'.
