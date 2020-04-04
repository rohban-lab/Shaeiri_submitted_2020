
### This code contains the adversarial training of cifar on 32/255 from madry model.


import torch

import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim


transform = transforms.Compose([transforms.ToTensor()])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                      download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                          shuffle=True, num_workers=1)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                     download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                         shuffle=False, num_workers=1)

from robustness.datasets import CIFAR
from robustness.model_utils import make_and_restore_model


ds = CIFAR('./data/cifar-10-batches-py')
model, _ = make_and_restore_model(arch='resnet50', dataset=ds, resume_path='./cifar_linf_8.pt?dl=0')
model.eval()
net = model


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)


def pgd_attack(model, images, labels, eps=32/255, alpha=2/255, iters=40):
    
    images = images.to(device)
    labels = labels.to(device)

    loss = torch.nn.CrossEntropyLoss()
    
    ori_images = images
    for i in range(iters):
        
        images.requires_grad = True
        outputs = model(images)
        model.zero_grad()

        cost = loss(outputs[0], labels).to(device)
        cost.backward()

        adv_images = images + alpha*images.grad.sign()
        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        images = torch.clamp(ori_images + eta, min=0, max=1).detach_()
        
    return images


num_epchos = 5

for epoch in range(num_epchos):
    
    steps = 0
    running_loss = 0.0
    net.train()

    for i, data in enumerate(trainloader, 0):
        
        inputs, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()
        
        adv = pgd_attack(net, inputs, labels)

        outputs = net(adv)
    
        loss = criterion(outputs[0], labels)
        loss.backward()

        optimizer.step()

        steps += 1
        running_loss += loss.item()

    # print statistics
    print('%d loss: %.5f' % (epoch + 1, running_loss / steps))

    net.eval()

    correct = 0
    total = 0
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        adv = pgd_attack(net, images, labels)
        outputs = net(adv)
        _, predicted = torch.max(outputs[0].data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %f %%' % (100 * correct / total))

    PATH = './cifar-32' + str(epoch + 1) + '.pth'
    torch.save(net.state_dict(), PATH)

print('Finished Training')


