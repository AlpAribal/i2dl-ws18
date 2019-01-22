from random import shuffle
import numpy as np

import torch
from torch.autograd import Variable


class Solver(object):
    default_adam_args = {"lr": 1e-4,
                         "betas": (0.9, 0.999),
                         "eps": 1e-8,
                         "weight_decay": 0.0}

    def __init__(self, optim=torch.optim.Adam, optim_args={},
                 loss_func=torch.nn.CrossEntropyLoss()):
        optim_args_merged = self.default_adam_args.copy()
        optim_args_merged.update(optim_args)
        self.optim_args = optim_args_merged
        self.optim = optim
        self.loss_func = loss_func

        self._reset_histories()

    def _reset_histories(self):
        """
        Resets train and val histories for the accuracy and the loss.
        """
        self.train_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        self.val_loss_history = []

    def train(self, model, train_loader, val_loader, num_epochs=10, log_nth=0, fresh=True):
        """
        Train a given model with the provided data.

        Inputs:
        - model: model object initialized from a torch.nn.Module
        - train_loader: train data in torch.utils.data.DataLoader
        - val_loader: val data in torch.utils.data.DataLoader
        - num_epochs: total number of training epochs
        - log_nth: log training accuracy and loss every nth iteration
        """
        iter_per_epoch = len(train_loader)
        if fresh == True:
            optim = self.optim(model.parameters(), **self.optim_args)    
            self.alp = optim        
            self._reset_histories()
            #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            device = "cpu"
            model.to(device)
        else:
            optim = self.alp

        print('START TRAIN.')
        ########################################################################
        # TODO:                                                                #
        # Write your own personal training method for our solver. In each      #
        # epoch iter_per_epoch shuffled training batches are processed. The    #
        # loss for each batch is stored in self.train_loss_history. Every      #
        # log_nth iteration the loss is logged. After one epoch the training   #
        # accuracy of the last mini batch is logged and stored in              #
        # self.train_acc_history. We validate at the end of each epoch, log    #
        # the result and store the accuracy of the entire validation set in    #
        # self.val_acc_history.                                                #
        #                                                                      #
        # Your logging could like something like:                              #
        #   ...                                                                #
        #   [Iteration 700/4800] TRAIN loss: 1.452                             #
        #   [Iteration 800/4800] TRAIN loss: 1.409                             #
        #   [Iteration 900/4800] TRAIN loss: 1.374                             #
        #   [Epoch 1/5] TRAIN acc/loss: 0.560/1.374                            #
        #   [Epoch 1/5] VAL   acc/loss: 0.539/1.310                            #
        #   ...                                                                #
        ########################################################################
        
        total_iter = num_epochs * iter_per_epoch
        for epoch in range(num_epochs):
            running_loss = 0.0
            running_count = 0.0
            running_correct = 0.0
            for i, data in enumerate(train_loader, 0):
                # get the inputs
                inputs, labels = data

                if torch.cuda.is_available():
                    inputs = inputs.cuda()
 
                # zero the parameter gradients
                optim.zero_grad()
 
                # forward pass
                outputs = model.forward(inputs)
 
                criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
                loss = criterion(outputs, labels)

                # backward pass
                loss.backward()

                # optimize
                optim.step()

                # loss and accuracy related calculations 
                running_loss += loss.item()
                running_count += len(labels)
                _, predicted = torch.max(outputs.data, 1)
                running_correct += torch.sum(predicted == labels.data).numpy()

                # logging - iteration loss
                if log_nth > 0 and i % log_nth == 0:
                    print('[Iteration %d, %d] TRAIN loss: %.3f' %(epoch*iter_per_epoch + i + 1, total_iter, running_loss/running_count))
            
            # logging - epoch train loss & accuracy
            running_loss /= running_count
            self.train_loss_history.append(running_loss)
            train_acc = 1.0*running_correct/running_count 
            self.train_acc_history.append(train_acc)
            print("[Epoch %d/%d] TRAIN acc/loss: %.3f/%.3f" % (epoch+1, num_epochs, train_acc, running_loss))

            # logging - epoch validation loss & accuracy
            val_correct = 0.0
            val_count = 0.0
            val_loss = 0.0
            with torch.no_grad():
                for val_data in val_loader:
                    inputs, labels = val_data
                    outputs = model(inputs)

                    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
                    val_loss += criterion(outputs, labels).item()

                    _, predicted = torch.max(outputs.data, 1)
                    val_count += labels.size(0)
                    val_correct += (predicted == labels).sum().item()            

            val_loss /= val_count
            self.val_loss_history.append(val_loss)
            val_acc = 1.0*val_correct/val_count 
            self.val_acc_history.append(val_acc)
            print("[Epoch %d/%d] VAL acc/loss: %.3f/%.3f" % (epoch+1, num_epochs, val_acc, val_loss))

        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################
        print('FINISH.')
