# %%
# Modifications by @jaspock

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import math
import matplotlib.pyplot as plt

# softmax regression for the MNIST digits dataset

# adapted from https://github.com/yunjey/pytorch-tutorial

class SoftmaxRegressor(nn.Module):

    def __init__(self, input_size, num_classes):
        super().__init__()
        self.linear = nn.Linear(input_size, num_classes)
        # log_softmax does not have parameters, so it is not needed here
    
    def forward(self, X):
        return F.log_softmax(self.linear(X), dim=1)

# An equivalent alternative to the above class using modules instead of functions:
class SoftmaxRegressor2(nn.Module):
    
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.linear = nn.Linear(input_size, num_classes)
        self.ls = nn.LogSoftmax(dim=1) 
    
    def forward(self, X):
        return self.ls(self.linear(X))


if __name__ == "__main__":

    input_size = 28*28  # 784
    num_classes = 10
    num_epochs = 2
    batch_size = 100
    learning_rate = 0.001

    # DataLoader wraps an iterable around the Dataset:
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    torch.manual_seed(10)

    model = SoftmaxRegressor(input_size, num_classes)
    # torch.nn.CrossEntropyLoss if logits are passed as input instead of log probs
    # torch.nn.BCELoss for 2 classes
    # If we used CrossEntropyLoss, we wouldn't need to apply the log_softmax function in the forward pass,
    # and the index of the true class would be used as the target instead of a one-hot encoded tensor.
    criterion = nn.NLLLoss()   
    
    # basic stochastic gradient descent optimizer:
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  

    # training:

    model.train() # set dropout and batch normalization layers to train mode

    losses = []
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # reshape images to (batch_size, input_size):
            images = images.reshape(-1, input_size)
            # forward pass:
            # the dunder (double underscores) magic method __call__ is called
            # __call_ in Module calls the override forward method in the child class
            # images.shape[0] calls the magic method __getitem__ that allows operator overloading 
            outputs = model(images)
            # labels contain the true class indices, not one-hot encoded vectors:
            loss = criterion(outputs, labels)
            # backward and optimize:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i+1) % 100 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                    .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
                losses.append(loss.item())

    plt.plot(losses, label='Training loss')
    plt.xlabel('steps x 100')
    plt.ylabel('cross entropy loss')
    plt.title('Evolution of loss during training')
    plt.legend(loc="upper right")
    plt.show()
    # plt.savefig("training_loss.png")

    model.eval()  # set dropout and batch normalization layers to evaluation mode

    # compute accuracy over the test set:
    # .no_grad() disables gradient tracking, which reduces memory usage and speeds up computations
    # the result of any operation inside the with block will have requires_grad=False
    # .no_grad() is useful for inference, when you are sure that you will not call .backward()
    # it can be used as a context manager or decorator, and it is orthogonal to .eval() and .train()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.reshape(-1, input_size)
            outputs = model(images)
            predicted = torch.argmax(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum()

        accuracy = 100 * correct / total
        print(f'Accuracy of the model on the 10000 test images: {accuracy:.2f}')

    # save the model checkpoint:
    # state_dict is a dictionary mapping each layer to its parameter tensor
    # torch.save(model.state_dict(), 'model.pt')

    # Extended saving:
    # torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 
    # 'optimizer_state_dict': optimizer.state_dict(),'loss': ... }, path)

    columns, rows = 5, 5
    width, height = 10, 10  # inches
    fig = plt.figure(figsize=(width, height))  # default dpi: 100 pixels-per-inch

    model.eval()

    for i in range(columns*rows):
        with torch.no_grad():
            img, label = train_dataset[i]   # torch.Size([1, 28, 28]), int
            input = img.reshape(-1, input_size)  # torch.Size([1, 784])
            outputs = model(input)  # torch.Size([1, 10])
            logprob, predicted_class = torch.max(outputs, dim=1)  # one-element tensors, dimension to reduce
            digit_img = img.numpy().reshape(28, 28)  # equivalentyl, detach().cpu().numpy()
            subplot = fig.add_subplot(rows, columns, i+1)
            subplot.set_title(f'{predicted_class.item()}, p={math.exp(logprob.item()):.2f} [{label}]')  
            subplot.axis('off')
            subplot.imshow(digit_img, cmap='gray')

    fig.show()
    # fig.savefig("test_digits.png")

# %%
