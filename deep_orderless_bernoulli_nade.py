import sys, os
import theano
import theano.tensor as T
from collections import OrderedDict
import numpy, scipy
from matplotlib import pyplot as plt
import utils
floatX = 'float32'
constantX = utils.constantX

class DeepOrderlessBernoulliNADE(object):
    def __init__(self, state, data_engine, channel):
        print 'Init ', self.__class__
        self.random_seed = state.random_seed
        self.rng_numpy, self.rng_theano = utils.get_two_rngs(seed=self.random_seed)
        self.state = state
        self.data_engine = data_engine
        self.save_model_path = state.save_model_path
        self.channel = channel
        config_model = state.DeepOrderlessNADE
        self.n_visible = config_model.n_in
        self.n_hidden = config_model.n_hidden
        # number of hidden layers besides the first and last
        self.n_layers = config_model.n_layers
        self.hidden_act = config_model.hidden_act
        self.tied_weights = config_model.tied_weights
        self.use_mask = config_model.use_mask
        self.init_mean_field = config_model.init_mean_field
        self.cost_from_last = config_model.cost_from_last
        self.init_weights = config_model.init_weights
        self.center_v = config_model.center_v
        config_train = state.DeepOrderlessNADE.train
        self.valid_freq = config_train.valid_freq
        self.n_orderings = config_train.n_orderings
        self.sgd_type = config_train.sgd_type
        self.n_epochs = config_train.n_epochs
        self.minibatch_size = config_train.minibatch_size
        self.momentum = config_train.momentum
        self.lr = config_train.lr
        self.l2 = config_train.l2
        # number of variationa inference to do
        self.k = self.state.DeepOrderlessNADE.train.k
        self.verbose = config_train.verbose
        self.fine_tune_n_epochs = config_train.fine_tune.n_epochs
        self.fine_tune_activate = config_train.fine_tune.activate
        assert self.n_layers >= 1

        # used in training, save also to txt file.
        self.LL_valid_test = []

        # for dataset
        self.trainset,_, self.validset, _ self.testset, _ = utils.load_mnist()
        self.marginal = numpy.mean(numpy.concatenate(
            [self.trainset, self.validset],axis=0),axis=0)

        # for tracking the costs for both pretrain and finetune
        self.costs = []
        self.costs_steps = []

        # decrease learning rate
        self.lr_decrease = self.lr / self.n_epochs
        
    def build_theano_fn_nade_k_rbm(self):
        # this is the variational rbm version of NADE-K
        
        self.x = T.fmatrix('inputs')
        self.x.tag.test_value = numpy.random.binomial(n=1,p=0.5,
            size=(self.minibatch_size,self.n_visible)).astype(floatX)
        self.m = T.fmatrix('masks')
        self.m.tag.test_value = numpy.random.binomial(n=1,p=0.5,
            size=(self.minibatch_size,self.n_visible)).astype(floatX)
        t = self.trainset[:self.minibatch_size]
        self.x.tag.test_value = t
        self.m.tag.test_value = utils.generate_masks_deep_orderless_nade(
                    t.shape, self.rng_numpy)
        # params of first layer
        self.W1 = utils.build_weights(
            n_row=self.n_visible, n_col=self.n_hidden, style=self.init_weights,
            name='W1',rng_numpy=self.rng_numpy)
        
            
        self.Wflags = utils.build_weights(
            n_row=self.n_visible, n_col=self.n_hidden, style=self.init_weights,
            name='Wflags',rng_numpy=self.rng_numpy) 
        self.b1 = utils.build_bias(size=self.n_hidden, name='b_1')
        self.c = utils.build_bias(size=self.n_visible, name='c')
        if self.tied_weights:
            print 'W1 and V are tied'
            self.V = self.W1
            self.params = [self.W1, self.Wflags, self.b1, self.c]
        else:
            print 'W1 and V are untied'
            self.V = utils.build_weights(
            n_row=self.n_visible, n_col=self.n_hidden, style=self.init_weights,
            name='V',rng_numpy=self.rng_numpy)
            self.params = [self.W1, self.Wflags, self.b1, self.c, self.V]
        
        if self.n_layers == 2:
            self.W2 = utils.build_weights(
            n_row=self.n_hidden, n_col=self.n_hidden, style=self.init_weights,
            name='W2',rng_numpy=self.rng_numpy)
            self.b2 = utils.build_bias(size=self.n_hidden, name='b_2')
            self.params += [self.W2, self.b2]
            
            
        # (B,k,D)
        self.mf = theano.shared(
            value=numpy.zeros((self.minibatch_size,self.k,self.n_visible)).astype(floatX),
            name='mean_field_v')
        cost, costs_by_step = self.get_nade_k_rbm_cost_theano(self.x, self.m, self.k)
        
        #L2_cost = T.sum(self.W1**2)  + T.sum(self.Wflags**2)
        #reg_cost = cost + self.l2 * L2_cost
        for param in self.params:
            if param.ndim == 2:
                cost += T.sum(param**2) * constantX(self.l2)
        # get gradients
        self.learning_rate = theano.shared(numpy.float32(self.lr), name='learning_rate')
        
        
        updates = OrderedDict()
        consider_constant = None
        if self.sgd_type == 0:
            print 'use momentum sgd'
            which_type = 0
        elif self.sgd_type == 1:
            print 'use adadelta sgd'
            which_type = 1
        else:
            raise NotImplementedError()
        updates = utils.build_updates_with_rules(
            cost, self.params,
            consider_constant, updates,
            self.learning_rate, self.lr_decrease, self.momentum,
            floatX, which_type 
        )
        
        # compile training functions
        print 'compiling fns ...'
        self.train_fn = theano.function(
            inputs=(self.x, self.m),
            outputs=[cost, costs_by_step],
            updates=updates,
            name='train_fn'
        )

        self.sampling_fn = self.get_nade_k_rbm_sampling_fn_theano(self.k)
        self.compute_LL_with_ordering_fn = self.get_nade_k_rbm_LL_theano(self.k)
        # this is build later
        self.inpainting_fn = None
        
    def get_nade_k_rbm_cost_theano(self, x, input_mask, k):
        """
        log p(x_missing | x_observed)
        x is a matrix of column datapoints (mbxD)
        D = n_visible, mb = mini batch size
        """
        #x_ = utils.corrupt_with_salt_and_pepper(
        #    x, x.shape, self.noise, rng_theano)
        #BxD
        print 'building cost function ...'
        output_mask = constantX(1)-input_mask
        D = constantX(self.n_visible)
        d = input_mask.sum(1)
        cost = constantX(0)
        costs_by_step = []
        print 'do %d steps of mean field inference'%k
        P = self.get_nade_k_mean_field(x, input_mask, k)

        costs = []
        for i, p in enumerate(P):
            # Loglikelihood on missing bits
            lp = ((x*T.log(p) + (constantX(1)-x)*T.log(constantX(1)-p)) \
                  * output_mask).sum(1) * D / (D-d)
            this_cost = -T.mean(lp)
            costs.append(this_cost)
            costs_by_step.append(this_cost)
        costs_by_step = T.stack(costs_by_step)
        if not self.cost_from_last:
            cost = T.mean(T.stacklists(costs))
        else:
            cost = costs[-1]
        
        return cost, costs_by_step

    def get_nade_k_rbm_sampling_fn_theano(self, k):
        # give one sample from NADE-k
        # this is a not so-optimized version, running the full model each time
        ordering = T.ivector('ordering')
        ordering.tag.test_value = range(self.W1.get_value().shape[0])
        samples_init = theano.tensor.constant(numpy.zeros((self.n_visible,),dtype=floatX))
        # [0,1,0,0,1,0] where 1 indicates bits that are observed
        input_mask_init = theano.tensor.constant(numpy.zeros((self.n_visible,),dtype=floatX))
        def sample_one_bit(
                this_bit,   # the idx in the ordering that is sampled this time
                sampled,    # [x1, 0, 0, x4, 0, 0, x7] with some bits already sampled
                input_mask, # [1,  0, 0 ,1,  0, 0, 1 ] with 1 indicates bits already sampled
                W1,Wflags,c):
            one = theano.tensor.constant(1, dtype=floatX)
            # [0,0,0,1,0,0,0] where 1 indicates bits that mean field is trying to predict
            output_mask = T.zeros_like(input_mask)
            output_mask = T.set_subtensor(output_mask[this_bit], one)
            means = self.get_nade_k_mean_field(sampled, input_mask, k)
            # use the mean coming from the last step of mean field
            use_mean = means[-1]
            bit = self.rng_theano.binomial(p=use_mean,n=1,size=use_mean.shape,dtype=floatX)
            new_sample = sampled * input_mask + output_mask * bit
            # set the new input mask
            input_mask = T.set_subtensor(input_mask[this_bit], one)
            return new_sample, input_mask
        
        [samples, input_mask], updates = theano.scan(
            fn=sample_one_bit,
            outputs_info=[samples_init, input_mask_init],
            sequences=ordering,
            non_sequences=[self.W1,self.Wflags,self.c],
        )
        
        sample = samples[-1][T.argsort(ordering)]
        f = theano.function(
            inputs=[ordering],
            outputs=sample,
            updates=updates, name='nade_k_sampling_fn'
        )
        return f
    
    def get_nade_k_LL_ensemble_theano(self, k, n_orderings):
        # 1/M sum_M log (sum_K 1/k p(x_m | o_k))
        # only support matrix x with first dim 1
        ordering = T.imatrix('ordering')
        # (O,D)
        ordering.tag.test_value = numpy.repeat(
            numpy.arange(self.n_visible)[numpy.newaxis,:],n_orderings, axis=0).astype('int32')
        # (O,D)
        #input_mask_init = T.fmatrix('input_mask')
        #input_mask_init.tag.test_value = numpy.zeros((10,self.n_visible),dtype=floatX)
        input_mask_init = constantX(numpy.zeros((n_orderings,self.n_visible),dtype=floatX))
        x = T.fmatrix('samples')
        x.tag.test_value = numpy.random.binomial(n=1,
                            p=0.5,size=(1,self.n_visible)).astype(floatX)
        
        def compute_LL_one_column(
                this_bit_vector,   # vector
                input_mask, # [1,  0, 0 ,1,  0, 0, 1 ] with 1 indicates bits already sampled
                x,    # testset minibatches
                W1,Wflags,c):
            one = theano.tensor.constant(1, dtype=floatX)
            # a list of (k,O,D)
            x_ = T.addbroadcast(x,0)
            means = self.get_nade_k_mean_field(x_, input_mask, k)
            # use the mean coming from the last step of mean field
            # (O,D)
            use_mean = means[-1]
            mean_column = use_mean[T.arange(use_mean.shape[0]), \
                                    this_bit_vector]*constantX(0.9999)+ \
                                    constantX(0.0001*0.5)
            x_column = x.flatten()[this_bit_vector]
            LL = x_column*T.log(mean_column) + \
                   (constantX(1)-x_column)*T.log(constantX(1)-mean_column)
            # set the new input mask: (O,D)
            input_mask = T.set_subtensor(input_mask[T.arange(input_mask.shape[0]),
                                                    this_bit_vector],one)
            return LL, input_mask
        
        [LLs, input_mask], updates = theano.scan(
            fn=compute_LL_one_column,
            outputs_info=[None, input_mask_init],
            sequences=[ordering.T],
            non_sequences=[x, self.W1,self.Wflags,self.c],
        )
        # LLs: (D,O)
        LL = utils.log_sum_exp_theano(LLs.sum(axis=0),axis=-1) - T.log(ordering.shape[1])
        f = theano.function(
            inputs=[x, ordering],
            outputs=LL,
            updates=updates, name='LL_on_one_example_fn'
        )
        return f
    
    def get_nade_k_rbm_LL_theano(self, k):
        ordering = T.ivector('ordering')
        ordering.tag.test_value = range(self.W1.get_value().shape[0])
        # [0,1,0,0,1,0] where 1 indicates bits that are observed
        input_mask_init = theano.tensor.constant(numpy.zeros((
                                    self.n_visible,),dtype=floatX))
        x = T.fmatrix('samples')
        x.tag.test_value = numpy.random.binomial(n=1,
                            p=0.5,size=(self.n_visible,self.minibatch_size)).astype(floatX)
        
        def compute_LL_one_column(
                this_bit,   # the column idx in the ordering that LL is computed on 
                input_mask, # [1,  0, 0 ,1,  0, 0, 1 ] with 1 indicates bits already sampled
                x,    # testset minibatches
                W1,Wflags,c):
            
            one = theano.tensor.constant(1, dtype=floatX)
            # [0,0,0,1,0,0,0] where 1 indicates bits that mean field is trying to predict
            #output_mask = T.zeros_like(input_mask)
            #output_mask = T.set_subtensor(output_mask[this_bit], one)
            # x is (D,B)
            means = self.get_nade_k_mean_field(x.T, input_mask, k)
            # use the mean coming from the last step of mean field
            mean_column = means[-1][:, this_bit]*constantX(0.9999)+constantX(0.0001*0.5)
            x_column = x[this_bit,:]
            LL = x_column*T.log(mean_column) + \
                   (constantX(1)-x_column)*T.log(constantX(1)-mean_column)
            # set the new input mask
            input_mask = T.set_subtensor(input_mask[this_bit], one)
            return LL, input_mask
        # LLs (D,B)
        [LLs, input_mask], updates = theano.scan(
            fn=compute_LL_one_column,
            outputs_info=[None, input_mask_init],
            sequences=[ordering],
            non_sequences=[x, self.W1,self.Wflags,self.c],
        )
        log_likelihood = LLs.sum(axis=0)
        f = theano.function(
            inputs=[x, ordering],
            outputs=log_likelihood,
            updates=updates, name='nade_k_sampling_fn'
        )
        return f
    
    def get_nade_k_rbm_LL_theano_k_mixture(self, k):
        
        ordering = T.ivector('ordering')
        ordering.tag.test_value = range(self.W1.get_value().shape[0])
        # [0,1,0,0,1,0] where 1 indicates bits that are observed
        input_mask_init = theano.tensor.constant(numpy.zeros((
                                    self.n_visible,),dtype=floatX))
        x = T.fmatrix('samples')
        x.tag.test_value = numpy.random.binomial(n=1,
                            p=0.5,size=(self.n_visible,self.minibatch_size)).astype(floatX)
        
        def compute_LL_one_column(
                this_bit,   # the column idx in the ordering that LL is computed on 
                input_mask, # [1,  0, 0 ,1,  0, 0, 1 ] with 1 indicates bits already sampled
                x,    # testset minibatches
                W1,Wflags,c):
            
            one = theano.tensor.constant(1, dtype=floatX)
            # [0,0,0,1,0,0,0] where 1 indicates bits that mean field is trying to predict
            #output_mask = T.zeros_like(input_mask)
            #output_mask = T.set_subtensor(output_mask[this_bit], one)
            # x is (D,B)
            means = self.get_nade_k_mean_field(x.T, input_mask, k)
            # use the mean coming from the last step of mean field
            mean_column = [mean[:, this_bit]*constantX(0.9999)+constantX(0.0001*0.5) for mean in means]
            x_column = x[this_bit,:]
            LL=theano.tensor.zeros_like(x_column)
            for mean_column_ in mean_column:
                LL += x_column*T.log(mean_column_) + \
                   (constantX(1)-x_column)*T.log(constantX(1)-mean_column_)
                   
            # set the new input mask
            input_mask = T.set_subtensor(input_mask[this_bit], one)
            return LL, input_mask
        # LLs (D,B)
        [LLs, input_mask], updates = theano.scan(
            fn=compute_LL_one_column,
            outputs_info=[None, input_mask_init],
            sequences=[ordering],
            non_sequences=[x, self.W1,self.Wflags,self.c],
        )
        log_likelihood = LLs.sum(axis=0) / k
        f = theano.function(
            inputs=[x, ordering],
            outputs=log_likelihood,
            updates=updates, name='nade_k_sampling_fn'
        )
        return f
    '''
    ---------------------------------------------------------------------------------------
    '''    
    def build_theano_fn_deep_nade_1(self):
        print 'Build theano functions...'
        # build cost for deep nade
        self.learning_rate = theano.shared(numpy.float32(self.lr))
        self.x = T.fmatrix('inputs')
        self.x.tag.test_value = numpy.random.binomial(n=1,p=0.5,
            size=(self.minibatch_size,self.n_visible)).astype(floatX)
        self.m = T.fmatrix('masks')
        self.m.tag.test_value = numpy.random.binomial(n=1,p=0.5,
            size=(self.minibatch_size,self.n_visible)).astype(floatX)
        # params of first layer
        self.W1 = utils.build_weights(
            n_row=self.n_visible, n_col=self.n_hidden, style=1,
            name='W1',rng_numpy=self.rng_numpy)
        self.Wflags = utils.build_weights(
            n_row=self.n_visible, n_col=self.n_hidden, style=1,
            name='Wflags',rng_numpy=self.rng_numpy) 
        self.b1 = utils.build_bias(size=self.n_hidden, name='b_1')
        # params of the last layer
        self.V = utils.build_weights(
            n_row=self.n_visible, n_col=self.n_hidden, style=1,
            name='V',rng_numpy=self.rng_numpy)
        self.c = utils.build_bias(size=self.n_visible, name='c')
        # params of middle layers
        self.Ws = utils.build_weights(
            n_row=self.n_hidden, n_col=self.n_hidden, style=0,
            name='Ws',rng_numpy=self.rng_numpy,
            size=(self.n_layers,self.n_hidden, self.n_hidden))
        self.bs = utils.build_bias(size=(self.n_layers,self.n_hidden), name='bs')
        self.params = [self.W1, self.Wflags, self.b1, self.V, self.c, self.Ws, self.bs] 

        self.cost = self.get_deep_nade_1_cost_theano(self.x, self.m)
        
        L2_cost = T.sum(self.W1**2) + T.sum(self.V**2) + \
          T.sum(self.Ws**2) + T.sum(self.Wflags**2)
        self.reg_cost = self.cost + self.l2 * L2_cost 
        # get gradients
        updates = OrderedDict()
        consider_constant = None
        gparams = T.grad(self.reg_cost, self.params, consider_constant)
        # build momentum
        gparams_mom = []
        for param in self.params:
            gparam_mom = theano.shared(
                numpy.zeros(param.get_value(borrow=True).shape,
                dtype=floatX))
            gparams_mom.append(gparam_mom)
                    
        for gparam, gparam_mom, param in zip(gparams, gparams_mom, self.params):
            inc = self.momentum * gparam_mom - self.lr * gparam
            updates[gparam_mom] = inc
            updates[param] = param + inc

        # compile training functions
        print 'compiling fns ...'
        self.train_fn = theano.function(
            inputs=(self.x, self.m),
            outputs=self.reg_cost,
            updates=updates,
            name='train_fn'
        )
        # for batch gradient descent
        self.compute_gradient_fn = theano.function(
            inputs=[self.x, self.m],
            outputs=[self.cost] + map(T.as_tensor_variable, gparams),
            name='compute_grad_for_bgd_fn'
            )
        # for LBFGS
        self.cost_fn = theano.function(
            inputs=[self.x, self.m],
            outputs=self.cost,
            name='lbfgs_cost_fn'
            )
        
        self.compute_LL_with_ordering_fn = self.get_deep_nade_1_LL_theano()
        self.sampling_fn = self.get_deep_nade_1_sampling_fn_theano()
    
    
    def get_deep_nade_1_LL_theano(self):
        # the slow sampling procedure, with a fixed ordering
        ordering = T.ivector('ordering')
        ordering.tag.test_value = numpy.arange(self.n_visible).astype('int32')
        x = T.fmatrix('samples')
        x.tag.test_value = numpy.random.binomial(n=1,
                            p=0.5,size=(self.n_visible,10)).astype(floatX)
        n_h1 = self.W1.get_value().shape[1]
        a = T.alloc(numpy.float32(0),x.shape[1], n_h1) + self.b1
        def LL_one_bit(this_bit, a, x, W1, Wflags, Ws, bs, V, c):
            act_h1 = apply_act(a, self.hidden_act)
            act = act_h1
            for i in range(self.n_layers):
                act = apply_act(T.dot(act,Ws[i]) + bs[0], self.hidden_act)
            t = T.dot(act, V[this_bit]) + c[this_bit]
            mean = T.nnet.sigmoid(t)*0.9999 + 0.0001 * 0.5
            x_i = x[this_bit,:]
            LL = x_i * T.log(mean) + (1-x_i) * T.log(1-mean)
            new_a = a + T.dot(x_i.dimshuffle(0,'x'),W1[this_bit].dimshuffle('x',0)) \
                    + Wflags[this_bit]
            return LL, new_a
        
        [LLs, a], updates = theano.scan(
            fn=LL_one_bit,
            outputs_info=[None,a],
            sequences=ordering,
            non_sequences=[x,self.W1,self.Wflags,self.Ws,self.bs,self.V,self.c]
        )
        LLs = T.sum(LLs,axis=0)
        fn = theano.function(
            inputs=[x, ordering],
            outputs=LLs,
            updates=updates,
            name='logdensity_with_a_certain_ordering_fn'
        )
        return fn
    
    def get_deep_nade_1_cost_theano(self, x, mask):
        """
        log p(x_missing | x_observed)
        x is a matrix of column datapoints (mbxD)
        D = n_visible, mb = mini batch size
        """
        #BxD
        print 'building cost function ...'
        output_mask = constantX(1)-mask
        D = constantX(self.n_visible)
        #d is the 1-based index of the dimension whose value
        #to infer (not the size of the context)
        d = mask.sum(1) 
        masked_input = x * mask #BxD
        h = apply_act(T.dot(masked_input, self.W1) \
                                + T.dot(mask, self.Wflags)
                                 + self.b1, act=self.hidden_act) #BxH
        for l in xrange(self.n_layers):
            #BxH
            h = apply_act(T.dot(h, self.Ws[l]) +
                                    self.bs[l], act=self.hidden_act)
        t = T.dot(h, self.V.T) + self.c #BxD
        # the output has to be sigmoid, though hiddens acts are flexible
        # BxD        
        p_x_is_one = T.nnet.sigmoid(t) * constantX(0.9999) + \
          constantX(0.0001 * 0.5)
        lp = ((x*T.log(p_x_is_one) +
               (constantX(1)-x)*T.log(constantX(1)-p_x_is_one)) \
               * output_mask).sum(1) * D / (D-d) #B

        cost = T.mean(utils.constantX(-1) * lp)
        return cost

    
    
    def get_deep_nade_1_sampling_fn_theano(self):        
        # give one sample
        ordering = T.ivector('ordering')
        ordering.tag.test_value = range(self.W1.get_value().shape[0])
        a = T.alloc(numpy.float32(0)) + self.b1
        def sample_one_bit(this_bit, a,
                           W1,Wflags,
                           Ws,bs,V,c):
            act = apply_act(a, self.hidden_act)
            for i in range(self.n_layers):
                act = apply_act(T.dot(act,Ws[0]) + bs[0], self.hidden_act)
            preact = T.dot(act,V[this_bit]) + c[this_bit]
            mean = T.nnet.sigmoid(preact)*0.9999 + 0.0001 * 0.5
            bit = self.rng_theano.binomial(p=mean,n=1,size=mean.shape,dtype=floatX)
            new_a = a + bit * W1[this_bit] + Wflags[this_bit]
            return bit, new_a
        [samples, a], updates = theano.scan(
            fn=sample_one_bit,
            outputs_info=[None, a],
            sequences=ordering,
            non_sequences=[self.W1,self.Wflags,self.Ws,self.bs,self.V,self.c]
            
        )
        samples = samples[T.argsort(ordering)]
        f = theano.function(
            inputs=[ordering],
            outputs=samples,
            updates=updates, name='slow_sampling_fn'
        )
        return f

    def get_nade_k_mean_field(self, x, input_mask, k):
        # this procedure uses mask only at the first step of inference
        # x: all inputs (B,D)
        # input_mask: input masks (B,D)
        # output_mask: (B,D)
        # k: how many step of mf, int
        
        # the convergence is indicated by P
        P = []
        for i in range(k):
            if i == 0:
                # the first iteration of MeanField
                if self.init_mean_field:
                    v = x * input_mask + self.marginal * (1-input_mask)
                else:
                    v = x * input_mask
                    
                if self.use_mask:
                    print 'first step of inference uses masks'
                    #mask_as_inputs = 1-input_mask
                    mask_as_inputs = input_mask
                    #mask_as_inputs = 2*input_mask-1
                else:
                    print 'first step of inference does not use masks'
                    mask_as_inputs = T.zeros_like(input_mask)
            else:
                # the following iterations does not use mask as inputs
                if self.use_mask:
                    mask_as_inputs = input_mask
                else:
                    mask_as_inputs = T.zeros_like(input_mask)
            # mean field
            if self.center_v:
                print 'inputs are centered'
                v_ = v - self.marginal
            else:
                print 'inputs not centered'
                v_ = v
            h = apply_act(T.dot(v_, self.W1) \
                    + T.dot(mask_as_inputs, self.Wflags)
                    + self.b1, act=self.hidden_act)
            if self.n_layers == 2:
                h = apply_act(T.dot(h, self.W2)+self.b2,act=self.hidden_act)
                
            p_x_is_one = T.nnet.sigmoid(T.dot(h, self.V.T) + self.c)
            # to stabilize the computation
            p_x_is_one = p_x_is_one*constantX(0.9999) + constantX(0.0001 * 0.5)
            # v for the next iteration
            #v = x * input_mask + p_x_is_one * output_mask
            v = x * input_mask + p_x_is_one * (1-input_mask)
            P.append(p_x_is_one)
        return P
    
    def generate_samples_theano(self, n):
        # n: how many samples
        # theano version of slow sampling
        ordering = numpy.asarray(range(self.n_visible)).astype('int32')
        sampling_fn = self.sampling_fn
        samples = []
        for i in range(n):
            if self.verbose:
                sys.stdout.write('\rSampling %d/%d'%(i+1, n))
                sys.stdout.flush()
            sample = sampling_fn(ordering)
            samples.append(sample)
        # (n,D)
        samples = numpy.asarray(samples)
        return samples
            
    def estimate_log_LL_with_ordering(self, data):
        # for testing compute LL
        ordering = numpy.asarray(range(self.n_visible)).astype('int32')
        batches = data.reshape((10,1000,784))
        LLs_all = []
        for k in range(self.n_orderings):
            numpy.random.shuffle(ordering)
            LLs = []
            for i, batch in enumerate(batches):
                if self.verbose:
                    sys.stdout.write('\rComputing LL %d/%d'%(
                                 i, batches.shape[0]))
                    sys.stdout.flush()
                test_LL = self.compute_LL_with_ordering_fn(batch.T, ordering)
                LLs.append(test_LL)
            LLs_all.append(numpy.mean(LLs))
            print 'this order ',numpy.mean(LLs), 'average ', numpy.mean(LLs_all)
        mean_over_orderings = numpy.mean(LLs_all)
        print 'LL ', mean_over_orderings
        return mean_over_orderings

    def estimate_log_LL_after_train_ensemble(self, k, data, n_orderings):
        n_orderings = n_orderings
        LL_after_train_fn = self.get_nade_k_LL_ensemble_theano(k, n_orderings)
        orderings = [numpy.random.permutation(numpy.arange(self.n_visible))
                     for i in range(n_orderings)]
        orderings = numpy.asarray(orderings).astype('int32')
        lls = []
        to_save = []
        for i, d in enumerate(data):
            if self.verbose:
                sys.stdout.write('\rComputing LL %d/%d'%(i+1, data.shape[0]))
                sys.stdout.flush()
           
            ll = LL_after_train_fn(d[numpy.newaxis, :], orderings)
            lls.append(ll)
            if i % 10 == 0:
                print 'mean LL so far ',numpy.mean(lls)
                to_save.append([i, numpy.mean(lls)])
                numpy.savetxt(self.save_model_path+
                              'ensembled_test_LL_%d_orderings'%n_orderings, to_save)
        print 'LL ensemble %.2f'%numpy.mean(lls)
        
    def estimate_log_LL_after_train(self, k, data):
        
        LL_after_train_fn = self.get_nade_k_rbm_LL_theano(k)
        #LL_after_train_fn = self.get_nade_k_rbm_LL_theano_k_mixture(k)
        ordering = numpy.asarray(range(self.n_visible)).astype('int32')
        batches = data.reshape((10,1000,784))
        LLs_all = []
        for k in range(self.n_orderings):
            numpy.random.shuffle(ordering)
            LLs = []
            for i, batch in enumerate(batches):
                if self.verbose:
                    sys.stdout.write('\rComputing LL %d/%d'%(
                                 i, batches.shape[0]))
                    sys.stdout.flush()
                LL = LL_after_train_fn(batch.T, ordering)
                LLs.append(LL)
            LLs_all.append(numpy.mean(LLs))
            print 'this order ',numpy.mean(LLs), 'average ', numpy.mean(LLs_all)
        mean_over_orderings = numpy.mean(LLs_all)
        print 'LL ', mean_over_orderings

        return mean_over_orderings
        
    
    def inpainting(self,epoch, k):
        def compile_inpainting_fn(k):
            input_mask = self.m
            output_mask = constantX(1) - input_mask
            P = self.get_nade_k_mean_field(self.x, input_mask, k)
            P = T.stacklists(P)
            samples = self.rng_theano.binomial(n=1,p=P,size=P.shape, dtype=floatX)
            samples = samples * output_mask
            fn = theano.function(inputs=[self.x, self.m], outputs=samples,name='inpainting_fn')
            return fn
        if not self.inpainting_fn:
            self.inpainting_fn = compile_inpainting_fn(k)
        # generate a square
        input_mask = numpy.ones((28,28),dtype=floatX)
        input_mask[10:20, 10:20] = numpy.float32(0)
        input_mask = input_mask.flatten()
        output_mask = numpy.float32(1) - input_mask
        B = 10
        # inpainting how many time each
        N = 10
        xs = self.testset[:B]
        all_paints = []
        input_mask = input_mask[numpy.newaxis,:]
        for x in xs:
            x_paints = []
            x = x[numpy.newaxis,:]
            x_paints.append(x * input_mask)
            for n in range(N):
                # inpaint many times for one x
                x_mis = self.inpainting_fn(x, input_mask)
                inpainted = x_mis + x * input_mask
                a,b,c = inpainted.shape
                inpainted = inpainted.reshape((a*b,c))
                x_paints.append(inpainted)
            all_paints.append(numpy.concatenate(x_paints,axis=0))
        all_paints = numpy.concatenate(all_paints, axis=0)
        img = image_tiler.visualize_mnist(data=all_paints, how_many=all_paints.shape[0])
        save_path = self.save_model_path + 'inpainting_e%d.png'%epoch
        img.save(save_path)
        #os.system('eog %s'%save_path)
        
    def train_valid_test(self):
        
        # set visible bias, critical
        self.c.set_value(-numpy.log((1-self.marginal)/self.marginal).astype(floatX))
        #trainset = numpy.concatenate([self.trainset, self.validset],axis=0)
        self.simple_train_sgd(self.trainset, epoch=0, epoch_end=self.n_epochs)
        #self.simple_train_bgd(trainset.value, n_epochs=500)
        #self.simple_train_lbfgs(trainset.value, maxiter=1)
        
        if self.fine_tune_activate:
            # reset the learning rate
            print 'reset the learning rate for fine-tuning'
            self.learning_rate.set_value(numpy.float32(self.lr))
            self.lr_decrease = self.lr / self.fine_tune_n_epochs
            params = [param.get_value() for param in self.params]
            # set some hyperparams
            self.cost_from_last = True
            # rebuild all theano fns
            self.build_theano_fn_nade_k_rbm()
            # load old params
            assert len(self.params) == len(params)
            for param_new, param_old in zip(params, self.params):
                assert param_new.shape == param_old.get_value().shape
                param_old.set_value(param_new)
            epoch_start = self.n_epochs
            epoch_end = self.n_epochs + self.fine_tune_n_epochs
            print 'start fine tune training'
            self.simple_train_sgd(self.trainset, epoch_start, epoch_end)
            
    def simple_train_sgd(self, trainset, epoch, epoch_end):
        # train with SGD
        print 'Train %s with SGD'%self.__class__
        idx = range(trainset.shape[0])
        
        minibatch_idx_overall = utils.generate_minibatch_idx(
            trainset.shape[0], self.minibatch_size)
        while (epoch < epoch_end):
            costs_epoch = []
            costs_by_step_epoch = []
            for k, use_idx in enumerate(minibatch_idx_overall):
                if self.verbose:
                    sys.stdout.write('\rTraining minibatches %d/%d'%(
                             k, len(minibatch_idx_overall)))
                    sys.stdout.flush()
                minibatch_data = trainset[use_idx,:]
                minibatch_mask = utils.generate_masks_deep_orderless_nade(
                    minibatch_data.shape, self.rng_numpy)
                if 0:
                    # this is deep nade
                    cost = self.train_fn(minibatch_data, minibatch_mask)
                else:
                    # len(results)==2
                    results = self.train_fn(minibatch_data, minibatch_mask)
                    cost = results[0]
                    # results[1]: (1,k)
                    costs_by_step = results[1].flatten()
                costs_epoch.append(cost)
                costs_by_step_epoch.append(costs_by_step)
            # now linearly decrease the learning rate
            current_lr = self.learning_rate.get_value()
            new_lr = current_lr - numpy.float32(self.lr_decrease) 
            self.learning_rate.set_value(new_lr)
            cost_epoch_avg = numpy.mean(costs_epoch)
            cost_by_step_avg = numpy.asarray(costs_by_step_epoch).mean(axis=0)
            
            self.costs_steps.append(cost_by_step_avg)
            self.costs.append(cost_epoch_avg)
            print '\rTraining %d/%d epochs, cost %.2f, costs by step %s lr %.5f'%(
            epoch, epoch_end, cost_epoch_avg, numpy.round(cost_by_step_avg,2),current_lr)
            if epoch != 0 and (epoch+1) % self.valid_freq == 0:
                numpy.savetxt(self.save_model_path+'epoch_costs_by_step.txt',
                              self.costs_steps)
                numpy.savetxt(self.save_model_path+'epoch_costs.txt', self.costs)
                if self.channel:
                    self.channel.save()
                self.sample_nade_v0(epoch)
                self.make_plots(self.costs)
                self.visualize_filters(epoch)
                self.LL(epoch, save_nothing=False)
                self.inpainting(epoch, self.k)
                self.save_model(epoch)
            epoch += 1
        # end of training
        print

    def sample_nade_v0(self, epoch):
        # sample with a specific ordering from trained nade, the traditional way
        samples = self.generate_samples_theano(50)
        image_tiler.visualize_mnist(data=samples,
                                    save_path=self.save_model_path+'samples_i%d.png'%epoch,
                                    how_many=samples.shape[0])
        print
    def make_plots(self, costs):
        plt.plot(costs)
        plt.savefig(self.save_model_path+'costs.png')

    def LL(self, epoch, save_nothing=False):
        # post_train: indicate whether this is called in trained time or after trained
        
        print 'estimate LL on validset'
        valid_LL = self.estimate_log_LL_with_ordering(self.validset)
        print 'estimate LL on testset'
        test_LL = self.estimate_log_LL_with_ordering(self.testset)
        
        
        if not save_nothing:
            # this function is called during the training, will write disk a file
            self.LL_valid_test.append([epoch, valid_LL, test_LL])
            
        # for model selection and jobman
        t = numpy.asarray(self.LL_valid_test)
        best_idx = numpy.argmax(t[:,1])
        best_epoch = t[best_idx, 0]
        best_valid = t[best_idx, 1]
        test_ll = t[best_idx, 2]
        self.state['best_validset_LL'] = best_valid
        self.state['best_epoch'] = best_epoch
        self.state['test_LL'] = test_ll
        if self.channel:
            self.channel.save()
        print 'best valid LL %.2f at epoch %d, test LL is %.2f'%(best_valid,best_epoch,test_ll)

        if not save_nothing:
            numpy.savetxt(self.save_model_path+'valid_test_LL.txt', self.LL_valid_test)

    def visualize_filters(self, epoch):
        print 'saving filters'
        to_do = [self.W1.get_value(), self.V.get_value()]
        names = ['W1', 'V']
        for param, name in zip(to_do, names):
            filters = utils.visualize_first_layer_weights(param, [28,28])
            name = self.save_model_path + 'filters_e%d_%s.png'%(epoch,name)
            filters.save(name)
        
    def save_model(self, epoch):
        print 'saving model params'
        params = [param.get_value() for param in self.params]
        utils.dump_pkl(params, self.save_model_path + 'model_params_e%d.pkl'%epoch)

    def load_params(self, params_path):
        print '======================================'
        print 'loading learned parameters from %s'%params_path
        params = utils.load_pkl(params_path)
        assert len(self.params) == len(params)
        for param_new, param_old in zip(params, self.params):
            assert param_new.shape == param_old.get_value().shape
            param_old.set_value(param_new)

        print 'trained model loaded success!'
                    
def train_from_scratch(state, data_engine, channel=None):
    model = DeepOrderlessBernoulliNADE(state, data_engine, channel)
    #model.build_theano_fn_deep_nade_1()
    model.build_theano_fn_nade_k_rbm()
    model.train_valid_test()

def evaluate_trained(state, data_engine, params_file, channel=None):
    # extra set up for loaded models
    state.DeepOrderlessNADE.train.k = 5
    state.DeepOrderlessNADE.train.n_orderings=10
    state.DeepOrderlessNADE.center_v=False
    state.DeepOrderlessNADE.cost_from_last=True
    model = DeepOrderlessBernoulliNADE(state, data_engine, channel)
    model.build_theano_fn_nade_k_rbm()
    model.load_params(params_file)
    epoch = state.load_trained.epoch
    data,_ = model.data_engine.get_dataset(which='test', force=True)
    #k = state.DeepOrderlessNADE.train.k
    
    k=5
    model.estimate_log_LL_after_train(k, data)
    #model.estimate_log_LL_after_train_ensemble(k,data,n_orderings=128)
    
def continue_train(state, data_engine, params_file, channel=None):
    state.DeepOrderlessNADE.cost_from_last = True
    model = DeepOrderlessBernoulliNADE(state, data_engine, channel)
    model.build_theano_fn_nade_k_rbm()
    model.load_params(params_file)
    model.simple_train_sgd(model.trainset)
    
if __name__ == '__main__':
    train()
            
                
