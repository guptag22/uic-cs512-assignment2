import torch
import torch.optim as optim
import torch.utils.data as data_utils
from data_loader import get_dataset
import numpy as np
from crf import CRF
import time


# Tunable parameters
batch_size = 256
num_epochs = 10
max_iters  = 1000
print_iter = 25 # Prints results every n iterations
conv_shapes = [[1,64,128]] #


# Model parameters
input_dim = 128
embed_dim = 128
num_labels = 26
cuda = torch.cuda.is_available()

# Instantiate the CRF model
crf_model = CRF(input_dim, embed_dim, conv_shapes, num_labels, batch_size)

# Setup the optimizer
opt = optim.LBFGS(crf_model.parameters())


##################################################
# Begin training
##################################################
step = 0

# Fetch dataset
dataset = get_dataset()
split = int(0.5 * len(dataset.data)) # train-test split
train_data, test_data = dataset.data[:split], dataset.data[split:]
train_target, test_target = dataset.target[:split], dataset.target[split:]

# Convert dataset into torch tensors
train = data_utils.TensorDataset(torch.tensor(train_data).float(), torch.tensor(train_target).long())
test = data_utils.TensorDataset(torch.tensor(test_data).float(), torch.tensor(test_target).long())

# print(len(train[0][1][0]))
letterwise_train = []
letterwise_test = []
wordwise_train = []
wordwise_test = []
            
for i in range(num_epochs):
    print("Processing epoch {}".format(i))
    
    # Define train and test loaders
    train_loader = data_utils.DataLoader(train,  # dataset to load from
                                            batch_size=batch_size,  # examples per batch (default: 1)
                                            shuffle=True,
                                            sampler=None,  # if a sampling method is specified, `shuffle` must be False
                                            num_workers=5,  # subprocesses to use for sampling
                                            pin_memory=False,  # whether to return an item pinned to GPU
                                            )

    test_loader = data_utils.DataLoader(test,  # dataset to load from
                                        batch_size=batch_size,  # examples per batch (default: 1)
                                        shuffle=False,
                                        sampler=None,  # if a sampling method is specified, `shuffle` must be False
                                        num_workers=5,  # subprocesses to use for sampling
                                        pin_memory=False,  # whether to return an item pinned to GPU
                                        )
    print('Loaded dataset... ')

    # Now start training
    for i_batch, sample in enumerate(train_loader):

        train_X = sample[0]
        train_Y = sample[1]
        print(train_Y.dtype)

        if cuda:
            train_X = train_X.cuda()
            train_Y = train_Y.cuda()

        # compute loss, grads, updates:
        def closure() :
            opt.zero_grad() # clear the gradients
            tr_loss = crf_model.loss(train_X, train_Y) # Obtain the loss for the optimizer to minimize
            for name, param in crf_model.named_parameters():
                if param.requires_grad:
                    print (name, param.data)
            tr_loss.backward() # Run backward pass and accumulate gradients
            return tr_loss
        start = time.time()
        opt.step(closure) # Perform optimization step (weight updates)
        print("TIME ELAPSED = ", time.time() - start)
        # # print to stdout occasionally:
        # if step % print_iter == 0:
        #     random_ixs = np.random.choice(test_data.shape[0], batch_size, replace=False)
        #     test_X = test_data[random_ixs, :]
        #     test_Y = test_target[random_ixs, :]

        #     # Convert to torch
        #     test_X = torch.from_numpy(test_X).float()
        #     test_Y = torch.from_numpy(test_Y).long()

        #     if cuda:
        #         test_X = test_X.cuda()
        #         test_Y = test_Y.cuda()
        #     test_loss = crf_model.loss(test_X, test_Y)
        #     print(step, tr_loss.data, test_loss.data,
        #                tr_loss.data / len(train_X), test_loss.data / len(test_X))
        
        
			##################################################################
			# IMPLEMENT WORD-WISE AND LETTER-WISE ACCURACY HERE
			##################################################################
        
        # if step < 100 :
        #     random_ixs = np.random.choice(test_data.shape[0], batch_size, replace=False)
        #     test_X = test_data[random_ixs, :]
        #     test_Y = test_target[random_ixs, :]

        #     # Convert to torch
        #     test_X = torch.from_numpy(test_X).float()
        #     test_Y = torch.from_numpy(test_Y).long()

        #     if cuda:
        #         test_X = test_X.cuda()
        #         test_Y = test_Y.cuda()
        #     with torch.no_grad() :
        #         train_predictions = crf_model(train_X)
        #         test_predictions = crf_model(test_X)
        #     train_word_acc = 0
        #     train_letter_acc = 0
        #     for y,y_predict in zip(train_Y,train_predictions) :
        #         num_letters = torch.sum(y)                      ## Number of letters in the word
        #         if (torch.all(torch.eq(y[:num_letters], y_predict[:num_letters]))):      ## if all letters are predicted correct
        #             train_word_acc += 1
        #         train_letter_acc += num_letters - (((~torch.eq(y[:num_letters],y_predict[:num_letters])).sum()) / 2)
        #     test_word_acc = 0
        #     test_letter_acc = 0
        #     for y,y_predict in zip(test_Y,test_predictions) :
        #         num_letters = torch.sum(y)                      ## Number of letters in the word
        #         if (torch.all(torch.eq(y[:num_letters], y_predict[:num_letters]))):      ## if all letters are predicted correct
        #             test_word_acc += 1
        #         test_letter_acc += num_letters - (((~torch.eq(y[:num_letters],y_predict[:num_letters])).sum()) / 2)
        #     letterwise_train.append(train_letter_acc)
        #     letterwise_test.append(test_letter_acc)
        #     wordwise_train.append(train_word_acc)
        #     wordwise_test.append(test_word_acc)
        #     print("Training Accuracy : ")
        #     print("\tword accuracy = ",train_word_acc)
        #     print("\tletter accuracy = ",train_letter_acc)
        #     print("Test Accuracy : ")
        #     print("\tword accuracy = ",test_word_acc)
        #     print("\tletter accuracy = ",test_letter_acc)
            

        step += 1
        if step > max_iters: raise StopIteration

### TODO : plot letterwise accuracy for training and test using letterwise_train and letterwise_test

### TODO : plot wordwise accuracy for training and test using wordwise_train and wordwise_test