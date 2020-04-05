
## Essential Libraries.

import torch

import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F

from robustness.datasets import CIFAR
from robustness.model_utils import make_and_restore_model


## Reading Dataset.

transform = transforms.Compose([transforms.ToTensor()])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                      download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                          shuffle=True, num_workers=1)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                     download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                         shuffle=False, num_workers=1)


## Downloading and Loading Madry Lab Pre-Trained Model on CIFAR10.

# !wget "https://www.dropbox.com/s/yxn15a9zklz3s8q/imagenet_linf_8.pt?dl=0"

ds = CIFAR('./data/cifar-10-batches-py')
model, _ = make_and_restore_model(arch='resnet50', dataset=ds, resume_path='./cifar_linf_8.pt')
model.eval()
net = model


## Loading our Pre-Trained model on CIFAR10.

PATH = './cifar-32-5.pth'
net.load_state_dict(torch.load(PATH))
net.eval()


## PGD attack.

def pgd_attack(model, images, labels, eps=32/255, alpha=2/255):

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

        cost = loss(outputs[0], labels).to(device)
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
    _, predicted = torch.max(outputs[0].data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()
print('Adversarial Accuracy of the network on the 10000 test images: %f %%' % (100 * correct / total))


