"""Linear Softmax Classifier."""
# pylint: disable=invalid-name
import numpy as np

from .linear_classifier import LinearClassifier


def cross_entropoy_loss_naive(W, X, y, reg):
    """
    Cross-entropy loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # pylint: disable=too-many-locals
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    ############################################################################
    # TODO: Compute the cross-entropy loss and its gradient using explicit     #
    # loops. Store the loss in loss and the gradient in dW. If you are not     #
    # careful here, it is easy to run into numeric instability. Don't forget   #
    # the regularization!                                                      #
    ############################################################################
    D = W.shape[0]
    C = W.shape[1]
    N = X.shape[0]
    
    for n in range(N):
        # First, calculate the dot product between X[i] and W
        Xi_dot_W = np.zeros([C])
        for c in range(C):
            for d in range(D):
                Xi_dot_W[c] += X[n,d] * W[d,c]

        # Now calculate the exponentials
        # To prevent instability, remove the maximum of Xi_dot_W from each entry
        # Removing a constant term from the argument of exponential function
        # corresponds to multiplication and we do this for both the numerator 
        # and the denominator. So we are safe!
        # Note that instabilities may occur only when values are too large, not
        # when too small, as exp(t) tends to 0 as t tends to -inf.       
        max_Xi_dot_W = np.max(Xi_dot_W)
        exp_Xi_dot_W = np.empty([C])
        sum_exp_Xi_dot_W = 0.0
        for c in range(C):
            Xi_dot_W[c] -= max_Xi_dot_W
            exp_Xi_dot_W[c] = np.exp(Xi_dot_W[c])
            sum_exp_Xi_dot_W += exp_Xi_dot_W[c]
        
        y_hat = exp_Xi_dot_W / sum_exp_Xi_dot_W
        # Gradient
        for d in range(D):
            for c in range(C):
                dW[d,c] += y_hat[c] * X[n,d]
                if c == y[n]:
                   dW[d,c] -= X[n,d] 

        # Add to the loss
        loss += np.log(y_hat[y[n]])

    for c in range(C):
        for d in range(D):
            dW[d,c] = dW[d,c] / N

    # Calculate the regularization term
    # Note that W[:,0] is not added to the regularization term
    reg_loss = 0.0
    for c in range(C):
            for d in range(0,D):
                #   Loss
                if d > 0:
                    reg_loss += W[d, c] *  W[d, c]
                #   Gradient
                dW[d, c] += reg * W[d, c]                
    
    loss = -loss/N + 0.5 * reg * reg_loss
    ############################################################################
    #                          END OF YOUR CODE                                #
    ############################################################################

    return loss, dW


def cross_entropoy_loss_vectorized(W, X, y, reg):
    """
    Cross-entropy loss function, vectorized version.

    Inputs and outputs are the same as in cross_entropoy_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    ############################################################################
    # TODO: Compute the cross-entropy loss and its gradient without explicit   #
    # loops. Store the loss in loss and the gradient in dW. If you are not     #
    # careful here, it is easy to run into numeric instability. Don't forget   #
    # the regularization!                                                      #
    ############################################################################
    C = W.shape[1]
    N = X.shape[0]
    
    # Calculate dot product between X and W     
    X_dot_W = X.dot(W)
    # Now calculate the exponentials
    # To prevent instability, remove the maximum of Xi_dot_W from each entry
    # Removing a constant term from the argument of exponential function
    # corresponds to multiplication and we do this for both the numerator 
    # and the denominator. So we are safe!
    # Note that instabilities may occur only when values are too large, not
    # when too small, as exp(t) tends to 0 as t tends to -inf.   
    max_X_dot_W = np.max(X_dot_W, axis = 1)
    X_dot_W = (X_dot_W.T - max_X_dot_W).T
    exp_X_dot_W = np.exp(X_dot_W)
    # Calculate the sum of exponentials
    sum_X_dot_W = np.sum(exp_X_dot_W, axis = 1)
    # Calculate probabilities
    y_hat = (exp_X_dot_W.T / sum_X_dot_W).T
    # Calculate loss
    # Note that W[:,0] is not added to the regularization term
    loss = -np.sum(np.log(y_hat[range(N),y]) / N) + 0.5 * reg * np.sum(W[1:,:] * W[1:,:])
    # Calculate gradient
    y_hat[range(N), y] = y_hat[range(N), y] - 1
    dW = X.T.dot(y_hat)

    dW /= N
    dW += reg * W
    ############################################################################
    #                          END OF YOUR CODE                                #
    ############################################################################

    return loss, dW


class SoftmaxClassifier(LinearClassifier):
    """The softmax classifier which uses the cross-entropy loss."""

    def loss(self, X_batch, y_batch, reg):
        return cross_entropoy_loss_vectorized(self.W, X_batch, y_batch, reg)


def softmax_hyperparameter_tuning(X_train, y_train, X_val, y_val):
    # results is dictionary mapping tuples of the form
    # (learning_rate, regularization_strength) to tuples of the form
    # (training_accuracy, validation_accuracy). The accuracy is simply the
    # fraction of data points that are correctly classified.
    results = {}
    best_val = -1
    best_softmax = None
    all_classifiers = []
    learning_rates = [1e-7, 5e-7]
    regularization_strengths = [2.5e4, 5e4]

    ############################################################################
    # TODO:                                                                    #
    # Write code that chooses the best hyperparameters by tuning on the        #
    # validation set. For each combination of hyperparameters, train a         #
    # classifier on the training set, compute its accuracy on the training and #
    # validation sets, and  store these numbers in the results dictionary.     #
    # In addition, store the best validation accuracy in best_val and the      #
    # Softmax object that achieves this accuracy in best_softmax.              #                                      #
    #                                                                          #
    # Hint: You should use a small value for num_iters as you develop your     #
    # validation code so that the classifiers don't take much time to train;   # 
    # once you are confident that your validation code works, you should rerun #
    # the validation code with a larger value for num_iters.                   #
    ############################################################################
    # counter = 0
    # import time
    # range_lr = np.linspace(2e-6, 4e-6, 22)
    # range_reg = np.concatenate(([0.0], np.linspace(500, 1000, 7)))
    # tic = time.time()
    # print("Started at " + str(tic))
    # for l in range(len(range_lr)):
    #     last_acc_train = 0
    #     last_good_acc_val = 0
    #     last_counter = 0 # So that we see a print after each outer loop iteration         
    #     for r in range(len(range_reg)):
    #         lr = range_lr[l]
    #         reg = range_reg[r]
    #         counter = l * len(range_reg) + r + 1
    #         if counter >= last_counter + 10 or counter % 10 == 0:
    #             toc = time.time()
    #             print("Counter #" + str(counter) + ": Best validation accuracy so far is " + str(best_val) + ". Time per count: " + str((toc - tic)/counter))
    #         softmax = SoftmaxClassifier()
    #         softmax.train(X_train, y_train, learning_rate=lr, reg=reg, num_iters=3000,batch_size=200, verbose=False)
    #         y_train_pred = softmax.predict(X_train)
    #         accuracy_train = np.mean(y_train == y_train_pred)
    #         y_val_pred = softmax.predict(X_val)
    #         accuracy_val = np.mean(y_val == y_val_pred)
    #         results[(lr,reg)] = [accuracy_train, accuracy_val]
    #         all_classifiers.append(softmax)
    #         last_counter = counter
    #         if accuracy_val > best_val:
    #             best_val = accuracy_val
    #             best_softmax = softmax
    #             print("New best (" + str(accuracy_val) + "). LR: " + str(lr) + ", Reg: " + str(reg))
    #         # if accuracy_train < 0.35:
    #         #    print("Breaking because of low training accuracy (" + str(accuracy_train) + "). LR: " + str(lr) + ", Reg: " + str(reg))
    #         #    break
    #         # if last_good_acc_val > 1.2 * accuracy_val:
    #         #    print("Breaking because of decrease in validation accuracy. LR: " + str(lr) + ", Reg: " + str(reg))
    #         #    break
    #         last_acc_train = accuracy_train
    #         last_good_acc_val = max(last_good_acc_val, accuracy_val)
        
    #     print("Outer loop #" + str(l) + " completed at " + str(time.time()))
    best_val = -1
    best_train_acc = -1
    import time
    learning_rate = np.array([2.2e-6, 2.25e-6, 2.3e-6])
    reg = np.array([27, 29, 31, 33, 35])
    num_iters = 8500

    for lr in learning_rate:
        for r in reg:
            tic = time.time()
            softmax = SoftmaxClassifier()
            softmax.train(X_train, y_train, learning_rate=lr, reg=r, num_iters=num_iters,batch_size=200, verbose=True)
            y_train_pred = softmax.predict(X_train)
            accuracy_train = np.mean(y_train == y_train_pred)
            y_val_pred = softmax.predict(X_val)
            accuracy_val = np.mean(y_val == y_val_pred)
            if accuracy_val > best_val or (accuracy_val == best_val and accuracy_train > best_train_acc):
                best_val = accuracy_val
                best_train_acc = accuracy_train
                best_softmax = softmax
            print("LR: " + str(lr) + ", R: " + str(r) + ", NIter: " + \
                        str(num_iters) + ", Train_Acc: " + str(accuracy_train) + ", Val_Acc: " + str(accuracy_val) + ". Took " + str(time.time() - tic))

    ############################################################################
    #                              END OF YOUR CODE                            #
    ############################################################################
        
    # Print out results.
    for (lr, reg) in sorted(results):
        train_accuracy, val_accuracy = results[(lr, reg)]
        print('lr %e reg %e train accuracy: %f val accuracy: %f' % (
              lr, reg, train_accuracy, val_accuracy))
        
    print('best validation accuracy achieved during validation: %f' % best_val)

    return best_softmax, results, all_classifiers
