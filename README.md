This package contains the accompany code for the following paper:

Tapani Raiko, Li Yao, KyungHyun Cho, Yoshua Bengio  
[Iterative Neural Autoregressive Distribution Estimator (NADE-k)](http://arxiv.org/abs/1406.1485).   
Advances in Neural Information Processing Systems 2014 (NIPS14).

Setup
---------------------
#### Install Theano

Download Theano and make sure it's working properly.  
All the information you need can be found by following this link:  
http://deeplearning.net/software/theano/  
Make sure theano is added into your PYTHONPATH.

#### Install Jobman
Very detailed information can be found below:  
http://deeplearning.net/software/jobman/install.html.  
Make sure jobman is added into your PYTHONPATH.
 
#### Prepare the MNIST dataset
You can download the dataset from the links below.  
[trainset]
(http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_train.amat)  
[validset](http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_valid.amat)  
[testset](http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_test.amat)

After the dataset has been downloaded, make sure to change the <code>data_path</code> in <code>utils.py</code>.   

Reproducing the Results 
---------------------
#### Train the model
1. Change <code>exp_path</code> in <code>config.py</code>. This is the *directory* where all the training outputs are going to be placed. For different experiments, one needs to specify <code>'save_model_path'</code> in the same config file.
2. To run NADE-5 1HL in Table 1 of the paper, make sure <code>'n_layers': 1,</code> and <code>'l2': 0.0</code>.
3. To run NADE-5 2HL in Table 1 of the paper, make sure <code>'n_layers': 2,</code> and <code>'l2': 0.0012279827881</code>.

#### Evaluation

1h model: 
testset LL over 10 orderings = -89.43
testset LL over 128 ensembles = 

2h model:
testset LL over 10 orderings = -87.13

#### Contact

Questions? Contact us: li.yao@umontreal.ca