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
2. To run NADE-5 1HL in Table 1 of the paper, make sure   
<code>'n_layers': 1,</code> and <code>'l2': 0.0</code>.
3. To run NADE-5 2HL in Table 1 of the paper, make sure   
<code>'n_layers': 2,</code> and <code>'l2': 0.0012279827881</code>.
4. To start training, <code>python train_model.py</code>

It is highly recommended the code is run on GPUs. For how to make it happen, take a look at this place: http://deeplearning.net/software/theano/tutorial/using_gpu.html.

#### Training outputs
During the training, lots of information is printed out on the screen, and many files are written to the <code>save_mode_path</code>. You will be able to see the plot of drop of the training cost, the generated samples from the model, the log-likelihood on the validset and testset every <code>valid_freq</code> epochs.

If you use the default setup, the model will be pretrained for 1000 epochs, and finetuned for another 3000 epochs. To have a good generative model, one need to be patient :)

In addition, we have provided some training logs with which you should be able to match your experiments with. See in the directory <code>results</code>.

#### Evaluation
After training is done, it is time to get all those SOTA numbers in Table 1 of the paper. 
1. In <code>config.py</code>, change the option <code>'action'</code> to 1. Meanwhile make sure <code>'from_path'</code> points to the *directory* that contains <code>model_params_e*.pkl</code> and <code>model_configs.pkl</code>. The option <code>'epoch'</code> specify which model over there you would like to use.
2. Then <code>python train_model.py</code>
3. If all goes well, the evaluation script should be able to produce numbers that match those in the paper.

*You probably will be surprised when you see better numbers than those reported in our paper. Calm down and we know this could happen. The longer you train our model, the more likely you will get better numbers. And do spread your joy to us when this happens.*
 
1h model: 
testset LL over 10 orderings = -89.43
testset LL over 128 ensembles = 

2h model:
testset LL over 10 orderings = -87.13

#### Contact

Questions? Contact us: li.yao@umontreal.ca