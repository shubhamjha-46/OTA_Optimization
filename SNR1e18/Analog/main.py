from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
import torch
import math
import random

class Net(nn.Module): 
    def __init__(self): ## initialize by default for any object
        super(Net, self).__init__() ### initializes/inherites the aattributes and methods of parent classs (nn.module)
        self.conv1 = nn.Conv2d(1, 16, 3, 1) ## stride=1 (how much step you while scanning), nn.Conv2d will take in a 4D Tensor of inputChannel(for every batch among 64), OutputChannels=32,kernel size=3x3,Height x Width.
        #self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25) #prevents overfitting, drops a neuron with probability 0.25, i.e., assign wt=0 so that dimension unchanged. 
        self.dropout2 = nn.Dropout(0.5)
        #self.fc1 = nn.Linear(9216, 128)
        #self.fc2 = nn.Linear(128, 10)
        self.fc1=nn.Linear(2704, 10)

    def forward(self, x): ### class->methods->attribute. writing self implies your passing object to the method. static method, class method, static method
        x = self.conv1(x) ## first convolution step: taking element-wise products and then average. gives 64 x 32 x 26 x 26
        
        x = F.relu(x) ## activation step
        #x = self.conv2(x) #64x64x24x24
        #print (x.size())
        
        #x = F.relu(x)
        #print (x.size())a
        
        x = F.max_pool2d(x, 2) ## max pooling make 64 x 64 x 12 x 12
        #print (x.size())
        x = self.dropout1(x)
        #print (x.size())
        x = torch.flatten(x, 1) ##converts to single dimension neurons 64 x 9216
        #print (x.size())
        x = self.fc1(x) ## 64 x 128
        #print (x.size())
        #x = F.relu(x)
        #print (x.size())
        #x = self.dropout2(x)
        #print (x.size())
        #x = self.fc2(x)  ## 64 x 10
        #print (x.size())
        output = F.log_softmax(x, dim=1) 
        #print (x.size())
        #print (output[1].size())
        return output
	  
def train(args, model, device, train_loader, optimizer, epoch, trainloss, cn):
    model.train() 
    P=1
    SNR=1000000000000000000 #d=27210
    print('Analog: P={}, SNR={}'.format(P,SNR))
    for batch_idx, (data, target) in enumerate(train_loader): 
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad() 
        output = model(data) 
        loss = F.nll_loss(output, target)
        loss.backward()         
        if batch_idx>0:
            sum1=0
            for p in model.parameters():
                sum1+=p.grad.norm()**2       
            #print('Gradient norm after {} iterations is {}'.format(batch_idx,np.sqrt(sum1)))
            norm=np.sqrt(sum1)
            #print('Before',norm)
            for p in model.parameters():
                p.grad*=np.sqrt(P*27210)/np.sqrt(sum1)   
            sum1=0
            for p in model.parameters():
                sum1+=p.grad.norm()**2
            #print('After scaling:', np.sqrt(sum1))
            norm_grad=[]
            signs=[0]*4
            i=0
            for p in model.parameters():
                #print(p.grad.norm())
                sign=2*torch.randint(0,2,p.grad.size())-1
                signs[i]=sign
                i+=1
                var1=p.grad*sign
                #print(var1)
                var1=var1.reshape(torch.numel(p.grad))
                var1=var1.tolist()
                norm_grad=norm_grad+var1
            #print(norm_grad)
            rotated_norm_grad=sp.fwht(norm_grad)
            #print(rotated_norm_grad)
            rotated_norm_grad=torch.Tensor(rotated_norm_grad)/2**7.5
            #print(torch.norm(rotated_norm_grad))
            rotated_norm_grad=rotated_norm_grad.tolist()
                       
            
            
            
            '''Sending over the air'''
            r=2  ###subsampling
            indx=random.sample(range(27210),r)
            indx=np.sort(indx).tolist()
            sample_rot_grad=[rotated_norm_grad[i] for i in indx]
            sample_rot_grad=torch.Tensor(sample_rot_grad)
            y_transmitted=sample_rot_grad+np.sqrt(P/SNR)*torch.randn(r)
            
            
            sampled_rot_grad=[0]*27210
            for i in range(len(indx)):
                #print(Coded_Q_rot_norm_grad[i]==Q_rot_norm_grad[i])
                sampled_rot_grad[indx[i]]=y_transmitted[i]
            
            
            
            Q_rot=torch.Tensor(sampled_rot_grad)*2**7.5
            #print(Q_rot[indx[0]],Q_rot[indx[1]])
            Q_rot_norm_grad_hat=Q_rot.tolist()
            inv_q_rot_norm_grad=sp.ifwht(Q_rot)        
            inv_q_rot_norm_grad=torch.tensor(inv_q_rot_norm_grad, dtype=float)*(27210/len(indx))*norm/np.sqrt(P*27210)
            #inv_q_rot_norm_grad=inv_q_rot_norm_grad
            #print('After',torch.norm(inv_q_rot_norm_grad))
            print('Inverse rotation done.')
            s=[0,144, 160,27200,27210, np.inf]
            j=0
            for p in model.parameters():
                shape_grad=p.grad.size()
                #print(torch.numel(p.grad))
                var=inv_q_rot_norm_grad[s[j]:s[j+1]]
                #print(s[j], var)
                #var=var.reshape(torch.numel(p.grad))
                #var=var*QSGD(norm, p.grad, 165)
                #var.reshape(torch.numel(p.grad))            
                #temp=torch.zeros(torch.numel(p.grad)).to(dtype=torch.float32)            
                #print(p.grad)
                var=var.reshape(shape_grad).to(dtype=torch.float32)
                p.grad=var*signs[j]
                j+=1
                #print(p.grad.norm())
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.item()))        
        
        trainloss.append((1.0)*loss.item())
        np.savetxt("train_loss_anlg_highSNR_rot_RCD.csv", trainloss, delimiter =", ", fmt ='% s')   

def test(model, device, test_loader,accr):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accr.append(1. * correct / len(test_loader.dataset))
    np.savetxt("accuracy_anlg_highSNR_rot_RCD.csv", accr, delimiter =", ", fmt ='% s')
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=512, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 8)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=3, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args() 
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu") 

    train_kwargs = {'batch_size': args.batch_size} 
    test_kwargs = {'batch_size': args.test_batch_size} 
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs) 
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  
        ])
    dataset1 = datasets.MNIST('../../data', train=True, download=True,  
                       transform=transform)                          
    dataset2 = datasets.MNIST('../../data', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs) 
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net().to(device) 
    
    optimizer = optim.SGD(model.parameters(), lr=args.lr)  

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    trainloss=[]
    accr=[]
    for epoch in range(1, args.epochs + 1):        
        cn=0
        train(args, model, device, train_loader, optimizer, epoch, trainloss,cn)        
        test(model, device, test_loader,accr)
        scheduler.step()    
    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")
    channel_uses=[i*27210 for i in range(args.epochs)]
    plt.plot(channel_uses,accr)
    
    plt.xlabel('Channel uses')
    plt.ylabel('Training accuracy')
    plt.show()
    plt.plot(trainloss)
    plt.xlabel('#batches')
    plt.ylabel('loss')
    plt.show()
    

if __name__ == '__main__':
    main()
