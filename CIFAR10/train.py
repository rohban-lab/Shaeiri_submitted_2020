
## Essential Libraries.

import torch

import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F

from robustness.datasets import CIFAR
from robustness.model_utils import make_and_restore_model


## Hyperparameters.

FLAGS = {
"batch_size": 128,
"learning_rate" : 0.001,
"n_epoch": 5,
"eps": 32/255}


## Reading Dataset.

transform = transforms.Compose([transforms.ToTensor()])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                      download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=FLAGS["batch_size"],
                                          shuffle=True, num_workers=1)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                     download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=FLAGS["batch_size"],
                                         shuffle=False, num_workers=1)


## Downloading and Loading Madry Lab Pre-Trained Model on CIFAR10.

# !wget "https://www.dropbox.com/s/yxn15a9zklz3s8q/imagenet_linf_8.pt?dl=0"

ds = CIFAR('./data/cifar-10-batches-py')
model, _ = make_and_restore_model(arch='resnet50', dataset=ds, resume_path='./cifar_linf_8.pt')
model.eval()
net = model


## Loss and optimizer.

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), FLAGS["learning_rate"], momentum=0.9, weight_decay=5e-4)


## GPU!

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)


## PGD attack.

def pgd_attack(model, images, labels, eps=FLAGS["eps"], alpha=2/255):

    iters = int((2.5 * eps) / alpha)
    
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


## Extended Adversarial Training.

net.train()

num_epchos = FLAGS["n_epoch"]

for epoch in range(num_epchos):
    
    steps = 0
    running_loss = 0.0

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

print('Finished Training')


## Saving the model.

net.eval()

PATH = './cifar-'+ str(int(FLAGS["eps"] * 255)) '-' + str(FLAGS["n_epoch"]) + '.pth'
torch.save(net.state_dict(), PATH)


