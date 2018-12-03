"""Two Layer Network."""
# pylint: disable=invalid-name
import numpy as np


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network. The net has an input dimension
    of N, a hidden layer dimension of H, and performs classification over C
    classes. We train the network with a softmax loss function and L2
    regularization on the weight matrices. The network uses a ReLU nonlinearity
    after the first fully connected layer.

    In other words, the network has the following architecture:

    input - fully connected layer - ReLU - fully connected layer - softmax

    The outputs of the second fully-connected layer are the scores for each
    class.
    """

    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
        """
        Initialize the model. Weights are initialized to small random values and
        biases are initialized to zero. Weights and biases are stored in the
        variable self.params, which is a dictionary with the following keys:

        W1: First layer weights; has shape (D, H)
        b1: First layer biases; has shape (H,)
        W2: Second layer weights; has shape (H, C)
        b2: Second layer biases; has shape (C,)

        Inputs:
        - input_size: The dimension D of the input data.
        - hidden_size: The number of neurons H in the hidden layer.
        - output_size: The number of classes C.
        """
        self.params = {}
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def loss(self, X, y=None, reg=0.0):
        """
        Compute the loss and gradients for a two layer fully connected neural
        network.

        Inputs:
        - X: Input data of shape (N, D). Each X[i] is a training sample.
        - y: Vector of training labels. y[i] is the label for X[i], and each
          y[i] is an integer in the range 0 <= y[i] < C. This parameter is
          optional; if it is not passed then we only return scores, and if it is
          passed then we instead return the loss and gradients.
        - reg: Regularization strength.

        Returns:
        If y is None, return a matrix scores of shape (N, C) where scores[i, c]
        is the score for class c on input X[i].

        If y is not None, instead return a tuple of:
        - loss: Loss (data loss and regularization loss) for this batch of
          training samples.
        - grads: Dictionary mapping parameter names to gradients of those
          parameters  with respect to the loss function; has the same keys as
          self.params.
        """
        # pylint: disable=too-many-locals
        # Unpack variables from the params dictionary
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N, _ = X.shape

        # Compute the forward pass
        scores = None
        ########################################################################
        # TODO: Perform the forward pass, computing the class scores for the   #
        # input. Store the result in the scores variable, which should be an   #
        # array of shape (N, C).                                               #         
        ########################################################################
        # Perform layer 1 and layer 2 calculations
        # --- Layer 1: For each data point, ReLU(x_i) = max(x_i.w_1 + b_1, 0)
        layer1 = X.dot(W1) + b1
        layer1_zero_indices = layer1 < 0
        layer1[layer1_zero_indices] = 0
        # --- Layer 2: For each data point, Layer2(x_i) = x_i.w_2 + b_2
        scores = layer1.dot(W2) + b2
        ########################################################################
        #                              END OF YOUR CODE                        #
        ########################################################################

        # If the targets are not given then jump out, we're done
        if y is None:
            return scores

        # Compute the loss
        loss = None
        ########################################################################
        # TODO: Finish the forward pass, and compute the loss. This should     #
        # include both the data loss and L2 regularization for W1 and W2. Store#
        # the result in the variable loss, which should be a scalar. Use the   #
        # Softmax classifier loss. So that your results match ours, multiply   #
        # the regularization loss by 0.5                                       #
        ########################################################################
        # For numerical stability, subtract max of each row from every element 
        # of the row
        max_scores = np.max(scores, axis=1, keepdims=True)
        scores = scores - max_scores
        # Calculate exponentials
        exp_scores = np.exp(scores)
        # Calculate the sum of exponentials
        sum_exp_scores = np.sum(exp_scores, axis=1, keepdims=True)
        # Calculate class probabilities
        probs = exp_scores / sum_exp_scores
        # Calculate loss
        # Note that W1[:,0] is not added to the regularization term
        loss = -np.sum(np.log(probs[range(N),y]) / N) + 0.5 * reg * (np.sum(W1 * W1) + np.sum(W2 * W2))
        ########################################################################
        #                              END OF YOUR CODE                        #
        ########################################################################

        # Backward pass: compute gradients
        grads = {}
        ########################################################################
        # TODO: Compute the backward pass, computing the derivatives of the    #
        # weights and biases. Store the results in the grads dictionary. For   #
        # example, grads['W1'] should store the gradient on W1, and be a matrix#
        # of same size                                                         #
        ########################################################################
        # Gradient w.r.t W2
        probs[range(N), y] = probs[range(N), y] - 1
        probs /= N
        dW2 = layer1.T.dot(probs) + reg * W2
        grads["b2"] = np.sum(probs, axis=0)
        grads["W2"] = dW2

        # Gradient w.r.t W1
        # X.dot(layer1 > 0)
        # dW1 = np.sum(X, axis=1, keepdims=True).T
        # dW1 = np.repeat(dW1, repeats=W1.shape[1], axis=1)
        # dW1 = (layer1 > 0) * dW1
        # print(X)
        # print(layer1_zero_indices)
        dhidden = probs.dot(W2.T)
        dhidden[layer1_zero_indices] = 0
        grads["W1"] = X.T.dot(dhidden) + reg * W1
        grads["b1"] = np.sum(dhidden, axis=0)
        ########################################################################
        #                              END OF YOUR CODE                        #
        ########################################################################

        return loss, grads

    def train(self, X, y, X_val, y_val,
              learning_rate=1e-3, learning_rate_decay=0.95,
              reg=1e-5, num_iters=100,
              batch_size=200, verbose=False):
        """
        Train this neural network using stochastic gradient descent.

        Inputs:
        - X: A numpy array of shape (N, D) giving training data.
        - y: A numpy array f shape (N,) giving training labels; y[i] = c means
          that X[i] has label c, where 0 <= c < C.
        - X_val: A numpy array of shape (N_val, D) giving validation data.
        - y_val: A numpy array of shape (N_val,) giving validation labels.
        - learning_rate: Scalar giving learning rate for optimization.
        - learning_rate_decay: Scalar giving factor used to decay the learning
          rate after each epoch.
        - reg: Scalar giving regularization strength.
        - num_iters: Number of steps to take when optimizing.
        - batch_size: Number of training examples to use per step.
        - verbose: boolean; if true print progress during optimization.
        """
        # pylint: disable=too-many-arguments, too-many-locals
        num_train = X.shape[0]
        iterations_per_epoch = max(num_train // batch_size, 1)

        # Use SGD to optimize the parameters in self.model
        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for it in range(num_iters):
            X_batch = None
            y_batch = None

            ####################################################################
            # TODO: Create a random minibatch of training data and labels,     #
            # storing hem in X_batch and y_batch respectively.                 #
            ####################################################################
            indices = np.random.choice(num_train, batch_size, replace=True)
            X_batch = X[indices]
            y_batch = y[indices]
            ####################################################################
            #                             END OF YOUR CODE                     #
            ####################################################################

            # Compute loss and gradients using the current minibatch
            loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
            loss_history.append(loss)

            ####################################################################
            # TODO: Use the gradients in the grads dictionary to update the    #
            # parameters of the network (stored in the dictionary self.params) #
            # using stochastic gradient descent. You'll need to use the        #
            # gradients stored in the grads dictionary defined above.          #
            ####################################################################
            self.params["W2"] -= learning_rate * grads["W2"]
            self.params["b2"] -= learning_rate * grads["b2"]
            self.params["W1"] -= learning_rate * grads["W1"]
            self.params["b1"] -= learning_rate * grads["b1"]
            ####################################################################
            #                             END OF YOUR CODE                     #
            ####################################################################

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

            # Every epoch, check train and val accuracy and decay learning rate.
            if it % iterations_per_epoch == 0:
                # Check accuracy
                train_acc = (self.predict(X_batch) == y_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)

                # Decay learning rate
                learning_rate *= learning_rate_decay

        return {
            'loss_history': loss_history,
            'train_acc_history': train_acc_history,
            'val_acc_history': val_acc_history,
        }

    def predict(self, X):
        """
        Use the trained weights of this two-layer network to predict labels for
        data points. For each data point we predict scores for each of the C
        classes, and assign each data point to the class with the highest score.

        Inputs:
        - X: A numpy array of shape (N, D) giving N D-dimensional data points to
          classify.

        Returns:
        - y_pred: A numpy array of shape (N,) giving predicted labels for each
          of the elements of X. For all i, y_pred[i] = c means that X[i] is
          predicted to have class c, where 0 <= c < C.
        """
        y_pred = None

        ########################################################################
        # TODO: Implement this function; it should be VERY simple!             #
        ########################################################################
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']

        hidden_layer = X.dot(W1) + b1
        hidden_layer = np.maximum(0, hidden_layer)
        scores = hidden_layer.dot(W2) + b2
        y_pred = np.argmax(scores, axis=1)
        ########################################################################
        #                              END OF YOUR CODE                        #
        ########################################################################

        return y_pred


def neuralnetwork_hyperparameter_tuning(X_train, y_train, X_val, y_val):
    best_net = None # store the best model into this 

    ############################################################################
    # TODO: Tune hyperparameters using the validation set. Store your best     #
    # trained model in best_net.                                               #
    #                                                                          #
    # To help debug your network, it may help to use visualizations similar to #
    # the  ones we used above; these visualizations will have significant      #
    # qualitative differences from the ones we saw above for the poorly tuned  #
    # network.                                                                 #
    #                                                                          #
    # Tweaking hyperparameters by hand can be fun, but you might find it useful#
    # to  write code to sweep through possible combinations of hyperparameters #
    # automatically like we did on the previous exercises.                     #
    ############################################################################
    all_nets = []
    counter = 0
    best_val_acc = -1
    best_train_acc = -1
    import time
    import matplotlib.pyplot as plt
    # Constant params
    input_size = 32 * 32 * 3
    num_classes = 10
    num_iters = 6000

    # Params to test
    hidden_size = np.array([400])
    learning_rate = np.array([2.6e-3])
    learning_rate_decay = np.array([0.95])
    reg = np.array([.88, .89, .91])

    for hs in hidden_size:
        for lr in learning_rate:
            for lrd in learning_rate_decay:
                for r in reg:
                    tic = time.time()
                    net = TwoLayerNet(input_size, hs, num_classes)
                    stats = net.train(X_train, y_train, X_val, y_val,
                                num_iters=num_iters, batch_size=500,
                                learning_rate=lr, learning_rate_decay=lrd,
                                reg=r, verbose=False)

                    val_acc = (net.predict(X_val) == y_val).mean()
                    train_acc = (net.predict(X_train) == y_train).mean()
                    if val_acc > best_val_acc or (val_acc == best_val_acc and train_acc > best_train_acc):
                        best_val_acc = val_acc
                        best_train_acc = train_acc
                        best_net = net
                    print("HS: " + str(hs) + ", LR: " + str(lr) + ", LRD: " + str(lrd) + ", R: " + str(r) + ", NIter: " + \
                        str(num_iters) + ", Train_Acc: " + str(train_acc) + ", Val_Acc: " + str(val_acc) + ". Took " + str(time.time() - tic))
                    # Plot the loss function and train / validation accuracies
                    plt.subplots(nrows=2, ncols=1)

                    plt.subplot(2, 1, 1)
                    plt.plot(stats['loss_history'])
                    plt.title('Loss history')
                    plt.xlabel('Iteration')
                    plt.ylabel('Loss')

                    plt.subplot(2, 1, 2)
                    plt.plot(stats['train_acc_history'], label='train')
                    plt.plot(stats['val_acc_history'], label='val')
                    plt.title('Classification accuracy history')
                    plt.xlabel('Epoch')
                    plt.ylabel('Clasification accuracy')

                    plt.tight_layout()
                    plt.show()
                    all_nets.append(net)

                    # net = TwoLayerNet(input_size, 50, num_classes)
                    # stats = net.train(X_train, y_train, X_val, y_val,
                    #             num_iters=5000, batch_size=500,
                    #             learning_rate=1e-3, learning_rate_decay=0.95,
                    #             reg=0.5, verbose=True)
    print("Best valid acc: ", best_val_acc)
    ############################################################################
    #                               END OF YOUR CODE                           #
    ############################################################################
    return best_net
