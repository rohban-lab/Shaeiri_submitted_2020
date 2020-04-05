## Essential Libraries.

import torch

import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F


## Reading Dataset.

transform = transforms.Compose([transforms.ToTensor()])

trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                      download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                          shuffle=True, num_workers=1)

testset = torchvision.datasets.MNIST(root='./data', train=False,
                                     download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=10000,
                                         shuffle=False, num_workers=1)


## Model.

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, (5, 5), stride=1)
        self.pool = nn.MaxPool2d((2, 2), stride=2)
        self.conv2 = nn.Conv2d(32, 64, (5, 5), stride=1)
        self.fc1 = nn.Linear(1024, 1024)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 1024)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

net = Net()


## Loading our Pre-Trained model on MNIST.

PATH = './mnist-IAT-0.4.pth'
net.load_state_dict(torch.load(PATH))
net.eval()


## GPU!

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)


## PGD attack.

def pgd_attack(model, images, labels, eps=0.4, alpha=0.005):

    # iters = int((2.5 * eps) / alpha)
    iters = 200
    
    images = images.to(device)
    labels = labels.to(device)

    loss = torch.nn.CrossEntropyLoss()

    ori_images = images
    
    images = images + torch.zeros_like(images).uniform_(-eps, eps)
    images = torch.clamp(images, min=0, max=1)

    for i in range(iters):
        
        images.requires_grad = True

        outputs = model(images)

        model.zero_grad()

        cost = loss(outputs, labels).to(device)
        cost.backward()

        adv_images = images + alpha*images.grad.sign()
        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        images = torch.clamp(ori_images + eta, min=0, max=1).detach_()
        
    return images


## Test.

correct = 0
total = 0
for data in testloader:
    images, labels = data[0].to(device), data[1].to(device)
    adv = pgd_attack(net, images, labels)
    outputs = net(adv)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()
print('Adversarial Accuracy of the network on the 10000 test images: %f %%' % (100 * correct / total))


