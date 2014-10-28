import numpy
from PIL import Image
import os
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams
import cPickle
floatX = 'float32'
# This is where mnist dataset is found
data_path = '/data/lisatmp3/yaoli/datasets/mnist/sampled_version/'

def build_weights(n_row=None, n_col=None, style=None, name=None,
                  rng_numpy=None, value=None, size=None):
    # build shared theano var for weights
    if size is None:
        size = (n_row, n_col)
    if rng_numpy is None:
        rng_numpy, _ = get_two_rngs()
    if value is not None:
        print 'use existing value to init weights'
        if len(size) == 3:
            assert value.shape == (n_row, how_many, n_col)
        else:
            assert value.shape == (n_row, n_col)
        rval = theano.shared(value=value, name=name)
    else:
        if style == 0:
            # do this only when sigmoid act
            print 'init %s with FORMULA'%name
            value = numpy.asarray(rng_numpy.uniform(
                          low=-4 * numpy.sqrt(6. / (n_row + n_col)),
                          high=4 * numpy.sqrt(6. / (n_row + n_col)),
                          size=size), dtype=floatX)
        elif style == 1:
            print 'init %s with Gaussian (0, %f)'%(name, 0.01)
            value = numpy.asarray(rng_numpy.normal(loc=0, scale=0.01,
                                size=size), dtype=floatX)
        elif style == 2:
            print 'init with another FORMULA'
            value = numpy.asarray(rng_numpy.uniform(
                    low=-numpy.sqrt(6. / (n_row + n_col)),
                    high=numpy.sqrt(6. / (n_row + n_col)),
                    size=size), dtype=floatX)
        elif style == 3:
            print 'int weights to be all ones, only for test'
            value = numpy.ones(size, dtype=floatX)
        elif style == 4:
            print 'usual uniform initialization of weights -1/sqrt(n_in)'
            value = numpy.asarray(rng_numpy.uniform(
                low=-1/numpy.sqrt(n_row),
                high=1/numpy.sqrt(n_row), size=size), dtype=floatX)
        else:
            raise NotImplementedError()

        rval = theano.shared(value=value, name=name)
    return rval

def build_bias(size=None, name=None, value=None):
    # build theano shared var for bias
    if value is not None:
        assert value.shape == (size,)
        print 'use existing value to init bias'
        rval = theano.shared(value=value, name=name)
    else:
        rval = theano.shared(value=numpy.zeros(size, dtype=floatX), name=name)
    return rval

def create_dir_if_not_exist(directory):
    if not os.path.exists(directory):
        print 'creating directory %s'%directory
        os.makedirs(directory)
    else:
        print "%s already exists!"%directory

def get_two_rngs(seed=None):
    if seed is None:
        seed = 1234
    else:
        seed = seed
    rng_numpy = numpy.random.RandomState(seed)
    rng_theano = MRG_RandomStreams(seed)

    return rng_numpy, rng_theano

def load_pkl(path):
    """
    Load a pickled file.

    :param path: Path to the pickled file.

    :return: The unpickled Python object.
    """
    f = open(path, 'rb')
    try:
        rval = cPickle.load(f)
    finally:
        f.close()
    return rval

def dump_pkl(obj, path):
    """
    Save a Python object into a pickle file.
    """
    f = open(path, 'wb')
    try:
        cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)
    finally:
        f.close()
        
def generate_minibatch_idx(dataset_size, minibatch_size):
    # generate idx for minibatches SGD
    # output [m1, m2, m3, ..., mk] where mk is a list of indices
    assert dataset_size > minibatch_size
    n_minibatches = dataset_size / minibatch_size
    leftover = dataset_size % minibatch_size
    idx = range(dataset_size)
    if leftover == 0:
        minibatch_idx = numpy.split(numpy.asarray(idx), n_minibatches)
    else:
        print 'uneven minibath chunking, overall %d, last one %d'%(minibatch_size, leftover)
        minibatch_idx = numpy.split(numpy.asarray(idx)[:-leftover], n_minibatches)
        minibatch_idx = minibatch_idx + [numpy.asarray(idx[-leftover:])]
    minibatch_idx = [idx_.tolist() for idx_ in minibatch_idx]
    return minibatch_idx

def generate_masks_deep_orderless_nade(shape, rng_numpy):
    # to generate masks for deep orderless nade training
    """
    Returns a random binary maks with ones_per_columns[i] ones
    on the i-th column

    shape: (minibatch_size * n_dim)
    Example: random_binary_maks((3,5),[1,2,3,1,2])
    Out:
    array([[ 0.,  1.,  1.,  1.,  1.],
    [ 1.,  1.,  1.,  0.,  0.],
    [ 0.,  0.,  1.,  0.,  1.]])
    """
    ones_per_column = rng_numpy.randint(shape[1], size=shape[0])
    assert(shape[0] == len(ones_per_column))
    shape_ = shape[::-1]
    indexes = numpy.asarray(range(shape_[0]))
    mask = numpy.zeros(shape_, dtype="float32")
    for i,d in enumerate(ones_per_column):
        numpy.random.shuffle(indexes)
        mask[indexes[:d],i] = 1.0
    return mask.T

def constantX(value, float_dtype='float32'):
    """
    Returns a constant of value `value` with floatX dtype
    """
    return theano.tensor.constant(numpy.asarray(value, dtype=float_dtype))

def scale_to_unit_interval(ndar, eps=1e-8):
    """ Scales all values in the ndarray ndar to be between 0 and 1 """
    ndar = ndar.copy()
    ndar -= ndar.min()
    ndar *= 1.0 / (ndar.max() + eps)
    return ndar


def tile_raster_images(X, img_shape, tile_shape, tile_spacing=(0, 0),
                       scale_rows_to_unit_interval=True,
                       output_pixel_vals=True):
    """
    Transform an array with one flattened image per row, into an array in
    which images are reshaped and layed out like tiles on a floor.

    This function is useful for visualizing datasets whose rows are images,
    and also columns of matrices for transforming those rows
    (such as the first layer of a neural net).

    :type X: a 2-D ndarray or a tuple of 4 channels, elements of which can
    be 2-D ndarrays or None;
    :param X: a 2-D array in which every row is a flattened image.

    :type img_shape: tuple; (height, width)
    :param img_shape: the original shape of each image

    :type tile_shape: tuple; (rows, cols)
    :param tile_shape: the number of images to tile (rows, cols)

    :param output_pixel_vals: if output should be pixel values (i.e. int8
    values) or floats

    :param scale_rows_to_unit_interval: if the values need to be scaled before
    being plotted to [0,1] or not


    :returns: array suitable for viewing as an image.
    (See:`PIL.Image.fromarray`.)
    :rtype: a 2-d array with same dtype as X.

    """

    assert len(img_shape) == 2
    assert len(tile_shape) == 2
    assert len(tile_spacing) == 2

    # The expression below can be re-written in a more C style as
    # follows :
    #
    # out_shape    = [0,0]
    # out_shape[0] = (img_shape[0]+tile_spacing[0])*tile_shape[0] -
    #                tile_spacing[0]
    # out_shape[1] = (img_shape[1]+tile_spacing[1])*tile_shape[1] -
    #                tile_spacing[1]
    out_shape = [(ishp + tsp) * tshp - tsp for ishp, tshp, tsp
                        in zip(img_shape, tile_shape, tile_spacing)]
    
    if isinstance(X, tuple):
        assert len(X) == 4
        # Create an output numpy ndarray to store the image
        if output_pixel_vals:
            out_array = numpy.zeros((out_shape[0], out_shape[1], 4),
                                    dtype='uint8')
        else:
            out_array = numpy.zeros((out_shape[0], out_shape[1], 4),
                                    dtype=X.dtype)

        #colors default to 0, alpha defaults to 1 (opaque)
        if output_pixel_vals:
            channel_defaults = [0, 0, 0, 255]
        else:
            channel_defaults = [0., 0., 0., 1.]

        for i in xrange(4):
            if X[i] is None:
                # if channel is None, fill it with zeros of the correct
                # dtype
                dt = out_array.dtype
                if output_pixel_vals:
                    dt = 'uint8'
                out_array[:, :, i] = numpy.zeros(out_shape,
                        dtype=dt) + channel_defaults[i]
            else:
                # use a recurrent call to compute the channel and store it
                # in the output
                out_array[:, :, i] = tile_raster_images(
                    X[i], img_shape, tile_shape, tile_spacing,
                    scale_rows_to_unit_interval, output_pixel_vals)
        return out_array

    else:
        # if we are dealing with only one channel
        H, W = img_shape
        Hs, Ws = tile_spacing

        # generate a matrix to store the output
        dt = X.dtype
        if output_pixel_vals:
            dt = 'uint8'
        out_array = numpy.zeros(out_shape, dtype=dt)

        for tile_row in xrange(tile_shape[0]):
            for tile_col in xrange(tile_shape[1]):
                if tile_row * tile_shape[1] + tile_col < X.shape[0]:
                    this_x = X[tile_row * tile_shape[1] + tile_col]
                    if scale_rows_to_unit_interval:
                        # if we should scale values to be between 0 and 1
                        # do this by calling the `scale_to_unit_interval`
                        # function

                        this_img = scale_to_unit_interval(
                            this_x.reshape(img_shape))
                        
                    else:
                        this_img = this_x.reshape(img_shape)
                    # add the slice to the corresponding position in the
                    # output array
                    c = 1
                    if output_pixel_vals:
                        c = 255
                    out_array[
                        tile_row * (H + Hs): tile_row * (H + Hs) + H,
                        tile_col * (W + Ws): tile_col * (W + Ws) + W
                        ] = this_img * c
        return out_array
    
def visualize_mnist(data=None, save_path=None, how_many=None,
                    image_shape=None, display=False):
    assert how_many is not None, 'You have to specify how_many'
    
    if data is not None:
        train = data.astype('float32')
    else:
        raise NotImplementedError()
        train, _,_,_,_,_ = data_provider.load_mnist()
    design_matrix = train
    images = design_matrix[0:how_many, :]
    shape = int(numpy.ceil(numpy.sqrt(how_many)))
    #channel_length = 28 * 28
    if image_shape is None:
        image_shape = [28, 28]
    to_visualize = images
                    
    image_data = tile_raster_images(to_visualize,
                                    tile_shape=[shape,shape],
                                    img_shape=image_shape,
                                    tile_spacing=(2,2))
    im_new = Image.fromarray(numpy.uint8(image_data))
    if save_path is not None:
        print 'img saved to %s'%save_path
        im_new.save(save_path)
    if display:
        im_new.save('samples_mnist.png')
        os.system('eog samples_mnist.png')
    return im_new

def visualize_first_layer_weights(W, img_shape=None,dataset_name=None):
    imgs = W.T
    
    if dataset_name == 'MNIST':
        img_shape = [28,28]
        to_visualize = imgs
    else:
        raise NotImplementedError('%s does not support visulization of W'%self.dataset_name)

    t = int(numpy.ceil(numpy.sqrt(W.shape[1])))

    tile_shape = [t,t]

    img = visualize_weight_matrix_single_channel(img_shape,
                                                   tile_shape, to_visualize)
    return img

def shuffle_dataset(data):
    idx = shuffle_idx(data.shape[0])
    return data[idx], idx

def shuffle_idx(n, shuffle_seed=1234):
    print 'shuffling dataset'
    idx=range(n)
    numpy.random.seed(shuffle_seed)
    numpy.random.shuffle(idx)
    return idx

def load_mnist():
    #binarized_mnist_test.amat  binarized_mnist_train.amat  binarized_mnist_valid.amat
    print 'loading binary MNIST, sampled version'
    train_x = numpy.loadtxt(data_path + 'binarized_mnist_train.amat').astype('float32')
    valid_x = numpy.loadtxt(data_path + 'binarized_mnist_valid.amat').astype('float32')
    test_x = numpy.loadtxt(data_path + 'binarized_mnist_test.amat').astype('float32')
    train_y = numpy.zeros((train_x.shape[0],)).astype('int32')
    valid_y = numpy.zeros((valid_x.shape[0],)).astype('int32')
    test_y = numpy.zeros((test_x.shape[0],)).astype('int32')
    # shuffle dataset
    train_x, _ = shuffle_dataset(train_x)
    valid_x,_ = shuffle_dataset(valid_x)
    test_x,_ = shuffle_dataset(test_x)

    return train_x, train_y, valid_x, valid_y, test_x, test_y

def apply_act(x, act=None):
    # apply act(x)
    # linear:0, sigmoid:1, tanh:2, relu:3, softmax:4, ultra_fast_sigmoid:5
    if act == 'sigmoid' or act == 1:
        rval = T.nnet.sigmoid(x)
    elif act == 'tanh' or act == 2:
        rval = T.tanh(x)
    elif act == 'relu' or act == 3:
        rval = relu(x)
    elif act == 'linear' or act == 0:
        rval = x
    elif act == 'softmax' or act == 4:
        rval = T.nnet.softmax(x)
    elif act == 'ultra_fast_sigmoid' or act == 5:
        # does not seem to work with the current Theano, gradient not defined!
        rval = T.nnet.ultra_fast_sigmoid(x)
    else:
        raise NotImplementedError()
    return rval

def get_updates_grads_momentum(gparams, params, updates, lr, momentum, floatX):
    print 'building updates with momentum'
    # build momentum
    gparams_mom = []
    for param in params:
        gparam_mom = theano.shared(
            numpy.zeros(param.get_value(borrow=True).shape,
            dtype=floatX))
        gparams_mom.append(gparam_mom)

    for gparam, gparam_mom, param in zip(gparams, gparams_mom, params):
        inc = momentum * gparam_mom - lr * gparam
        updates[gparam_mom] = inc
        updates[param] = param + inc
    return updates

def get_updates_adadelta(grads, params, 
                         updates, learning_rate, lr_decrease, floatX, decay=0.95):
    decay = constantX(decay)
    def sharedX(value):
        return theano.shared(value)
    for param, grad in zip(params, grads):
        # mean_squared_grad := E[g^2]_{t-1}
        mean_square_grad = sharedX(numpy.zeros(param.get_value().shape, dtype=floatX))
        # mean_square_dx := E[(\Delta x)^2]_{t-1}
        mean_square_dx = sharedX(numpy.zeros(param.get_value().shape, dtype=floatX))
        if param.name is not None:
            mean_square_grad.name = 'mean_square_grad_' + param.name
            mean_square_dx.name = 'mean_square_dx_' + param.name

        # Accumulate gradient
        new_mean_squared_grad = \
                decay * mean_square_grad +\
                (1 - decay) * T.sqr(grad)

        # Compute update
        #epsilon = constantX(lr_decrease) * learning_rate
        epsilon = constantX(0.00001) * learning_rate
        #epsilon = constantX(1e-7)
        rms_dx_tm1 = T.sqrt(mean_square_dx + epsilon)
        rms_grad_t = T.sqrt(new_mean_squared_grad + epsilon)
        delta_x_t = - rms_dx_tm1 / rms_grad_t * grad

        # Accumulate updates
        new_mean_square_dx = \
                decay * mean_square_dx + \
                (1 - decay) * T.sqr(delta_x_t)

        # Apply update
        updates[mean_square_grad] = new_mean_squared_grad
        updates[mean_square_dx] = new_mean_square_dx
        updates[param] = param + delta_x_t
    return updates
            
def get_updates_Nesterov():
    pass

def get_updates_adagrad(cost, params, consider_constant, updates, lr, floatX):
    # based on
    # http://xcorr.net/2014/01/23/adagrad-eliminating-learning-rates-in-stochastic-gradient-descent/
    print 'building updates with AdaGrad'
    grads = T.grad(cost, params, consider_constant)
    fudge_factor = constantX(1e-6)
    grads_history = []
    for param in params:
        grad_h = theano.shared(numpy.zeros(param.get_value().shape, dtype=floatX))
        grads_history.append(grad_h)
    
    for grad, grad_h, param in zip(grads, grads_history, params):
        updates[grad_h] = grad_h + grad **2
        grad_adjusted = grad /(fudge_factor + T.sqrt(grad_h))
        updates[param] = param - lr * grad_adjusted
    return updates

def build_updates_with_rules(cost, params, consider_constant,
                             updates,
                             lr, lr_decrease, momentum,
                             floatX, which_type):
    # lr: scalar theano constant
    # lr_decrease: scalar python constant
    
    grads = T.grad(cost, params, consider_constant)
    if which_type == 0:
        # use easy SGD with momentum
        updates = get_updates_grads_momentum(
            grads, params, updates,
            lr, momentum, floatX)

    elif which_type == 2:
        # use adagrad
        updates = get_updates_adagrad(grads, params,
                updates, lr, floatX)

    elif which_type == 1:
        # use adadelta
        updates = get_updates_adadelta(grads, params, 
                updates, lr, lr_decrease, floatX)

    else:
        raise NotImplementedError()
    
    return updates

def log_sum_exp_theano(x, axis):
    max_x = T.max(x, axis)
    return max_x + T.log(T.sum(T.exp(x - T.shape_padright(max_x, 1)), axis))
