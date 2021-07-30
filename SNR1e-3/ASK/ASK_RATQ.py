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

class Net(nn.Module): ### nn.Module to use pre-defined classes like Conv2d
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
        
        #x = F.relu(x) ## activation step
        #x = self.conv2(x) #64x64x24x24
        #print (x.size())
        
        x = F.relu(x)
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
def unif_q(x,ind,M):
	K=[8,8,8,8]
	k=K[ind]
	I=M[ind]
	L=[-I+i*2*I/(k-1) for i in range(k)]   ## using 11-level uniform quantizer
	for u in range(len(L)):
		if L[u]<=x<=L[u+1]:
			if np.random.rand(1)<(k-1)*(x-L[u])/(2*I):
				return (L[u+1],u+1)
			else:
				return (L[u],u)
def quantizer(gr):
	e=0
	M=[0]*4
	d=2**15
	'''Adaptive intervals with h=log(1+ln*d/3)'''
	for t in range(4):
		M[t]=np.sqrt(3/d*np.exp(e))
		e=np.exp(e)
	#M=[1,1,1,1]
	if abs(gr)<M[0]:
		(q,w)=unif_q(gr,0,M)
		return (q,w,0,M)
	if M[0]<=abs(gr)<M[1]:
		(q,w)=unif_q(gr,1,M)
		return (q,w,1,M)
	if M[1]<=abs(gr)<M[2]:
		(q,w)=unif_q(gr,2,M)
		return (q,w,2,M)
	if M[2]<=abs(gr)<M[3]:
		(q,w)=unif_q(gr,3,M)
		return (q,w,3,M)
def RATQ(grad):
	#plt.hist(grad, bins=50)
	#plt.show()
	res=[0]*len(grad)
	Q_ind=[0]*len(grad)
	m_ind=[0]*len(grad)
	for i in range(len(grad)):
		(res[i],Q_ind[i],m_ind[i],M)=quantizer(grad[i])
	return (res,Q_ind,m_ind,M)
	  
def ASKencoder(P,l,idx):
	return (-np.sqrt(P)+2*idx*np.sqrt(P)/(2**l-1))

def ASKdecoder(P,y,l):
	if y<=-np.sqrt(P)+np.sqrt(P)/(2**l-1):
		return(0)
	elif y>= np.sqrt(P)-np.sqrt(P)/(2**l-1):
		return(2**l-1)
	else:
		#print(y, np.sqrt(P))
		#print(0.5+(y+np.sqrt(P))*(2**l-1)/2/np.sqrt(P))
		return(int(0.5+(y+np.sqrt(P))*(2**l-1)/2/np.sqrt(P)))
def train(args, model, device, train_loader, optimizer, epoch, trainloss, test_train_accr, test_train_loss,N,BS,cn):
    model.train() ## Just flags that model is in training mode; required by dropout()
    
    for batch_idx, (data, target) in enumerate(train_loader): ## enumerate use the batches available in size of 64.
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad() ## initialize the grad to 0.
        output = model(data)  ## creates a instance of the class
        loss = F.nll_loss(output, target)
        loss.backward() ## computes the gradients for model.parameters(); model.parameters has weights and biases for cnv1, cnv2, fc1, fc2 layers.
        #r=600
        #indx=random.sample(range(27210),int(r/5))
        #indx=np.sort(indx).tolist()
        #print(indx)
        if batch_idx>0:
            sum1=0
            s=[0,144, 160,27200,27210, np.inf]
            #s=[144,16,27040,10]
            #i=0
            for p in model.parameters():
                sum1+=p.grad.norm()**2
            norm=np.sqrt(sum1)
            #print('Before',norm)
            norm_grad=[]
            signs=[0]*4
            i=0
            for p in model.parameters():
                #print(p.grad.norm())
                sign=2*torch.randint(0,2,p.grad.size())-1
                signs[i]=sign
                i+=1
                var1=p.grad*sign/norm
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
            #print('Rotation done.', max(rotated_norm_grad))
            SNR=0.001
            nv=10000
            P=nv*SNR
            k=8           
            '''if SNR<1:
                r=int(np.log2(1+np.sqrt(4*SNR/np.log(2*N*60000/BS))))+1 
            else:
                r=int(np.log2(1+SNR/np.log(10)))## fixes ASK constellations'''
            
            r=int(np.log2(1+np.sqrt(4*SNR)))                         
            
            r=int(r/5)*5
            r=6
            print('P={}, SNR={}, r={}, Batch_id={}'.format(P,SNR,r,batch_idx))            
            
            
            indx=random.sample(range(27210),int(r/5))  ## 200 out of 27210
            indx=np.sort(indx).tolist()
            sample_rot_grad=[rotated_norm_grad[i] for i in indx]
            (Q_rot_norm_grad,Q_ind, m_ind,M)=RATQ(sample_rot_grad)
            #print('Quantization done.')
            #print('Before channel:',Q_rot_norm_grad)

            '''###########################'''
            
            ''' code before this point is RCD+raw RATQ'''
            '''Write your code here'''
            #Q_rot_norm_grad=[]
            m_ind_hat=[0]*len(indx)
            Q_ind_hat=[0]*len(indx)
            Coded_Q_rot_norm_grad=[0]*len(indx)
            m_ind_bin=''
            Q_ind_bin=''
            #print(Q_ind,m_ind)
            for i in range(len(indx)):
                m_ind_bin=m_ind_bin+bin(m_ind[i])[2:].zfill(2)  ###2 bits per coordinate....total of 200*2=400bits
                Q_ind_bin=Q_ind_bin+bin(Q_ind[i])[2:].zfill(3)  ###600 bits
            #print(Q_ind_bin, m_ind_bin)
            m_ASK_ind=int(m_ind_bin,2)    
            Q_ASK_ind=int(Q_ind_bin,2)
            #print(Q_ASK_ind,m_ASK_ind)
            e_m_ind=ASKencoder(P,2*len(indx),m_ASK_ind)
            e_Q_ind=ASKencoder(P,3*len(indx),Q_ASK_ind)
            y_m_ind=e_m_ind+np.sqrt(nv)*np.random.randn(1)
            y_Q_ind=e_Q_ind+np.sqrt(nv)*np.random.randn(1)
            #y_m_ind=e_m_ind
            #y_Q_ind=e_Q_ind
            #print(y_Q_ind,y_m_ind)
            cn+=2
            m_ASK_ind_hat=bin(ASKdecoder(P,y_m_ind,2*len(indx)))[2:].zfill(2*len(indx))
            Q_ASK_ind_hat=bin(ASKdecoder(P,y_Q_ind,3*len(indx)))[2:].zfill(3*len(indx))
            #print(Q_ASK_ind_hat,m_ASK_ind_hat)
            m_ind_hat=[int(m_ASK_ind_hat[i:i+2],2) for i in range(0,2*len(indx),2)]
            Q_ind_hat=[int(Q_ASK_ind_hat[i:i+3],2) for i in range(0,3*len(indx),3)]
            #print(Q_ind_hat,m_ind_hat)
            for i in range(len(indx)):
                Coded_Q_rot_norm_grad[i]=-M[m_ind_hat[i]]+Q_ind_hat[i]*2*M[m_ind_hat[i]]/(k-1)
            #Q_rot_norm_grad=[]
            Q_rot_norm_grad=Coded_Q_rot_norm_grad
            #print(Q_rot_norm_grad==Coded_Q_rot_norm_grad)
            #print('After channel:',Coded_Q_rot_norm_grad)
            '''############################'''
            '''Code after this point is RCD+raw RATQ'''
            sampled_rot_grad=[0]*27210
            for i in range(len(indx)):
                #print(Coded_Q_rot_norm_grad[i]==Q_rot_norm_grad[i])
                sampled_rot_grad[indx[i]]=Coded_Q_rot_norm_grad[i]
                           
            Q_rot=torch.tensor(sampled_rot_grad, dtype=float)*2**7.5
            #print(Q_rot[indx[0]],Q_rot[indx[1]])
            Q_rot_norm_grad_hat=Q_rot.tolist()
            inv_q_rot_norm_grad=sp.ifwht(Q_rot)        
            inv_q_rot_norm_grad=torch.tensor(inv_q_rot_norm_grad, dtype=float)*(27210/len(indx))*norm
            #inv_q_rot_norm_grad=inv_q_rot_norm_grad
            #print('After',torch.norm(inv_q_rot_norm_grad))
            print('Inverse rotation done.')
            j=0
            for p in model.parameters():
                shape_grad=p.grad.size()
                #print(torch.numel(p.grad))
                var=inv_q_rot_norm_grad[s[j]:s[j+1]]
                #print(var.size())
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
        print('Gradient update done with {} channel uses.'.format(cn))
        '''Notify the status after every fixed number of batches trained'''
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break
        trainloss.append((1.0)*loss.item())
        np.savetxt("train_loss_ASK_RATQ_SNR_1e04.csv", trainloss, delimiter =", ", fmt ='% s')
def test(model, device, test_loader,accr):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accr.append(1. * correct / len(test_loader.dataset))

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
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 1.0)')
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
    args = parser.parse_args() ###args has batch size, epochs, test batch size, etc. ###
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu") ##3 Uses cpu if not CUDA

    train_kwargs = {'batch_size': args.batch_size} ##dictionary keyword arguement
    test_kwargs = {'batch_size': args.test_batch_size} 
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs) 
        test_kwargs.update(cuda_kwargs)
    ##torchvision.transforms.ToTensor: Convert a PIL Image or numpy.ndarray to tensor. This transform does not support torchscript.
    ###Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  ## Normalize by subtracting mean 0.1307 for 1 channel and dividing by std. dev. 0.3081
        ])
    dataset1 = datasets.MNIST('../../data', train=True, download=True,  ## download the training dataset from internet if not already ##
                       transform=transform)                          
    dataset2 = datasets.MNIST('../../data', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs) ## loads and partition the dataset1 into batches and sets batch_size arguement using **train_kwargs
    #(reads train_kwargs dictionary defined above)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net().to(device) ## defining neural network model on torch.device(CPU or GPU if available)
    
    optimizer = optim.SGD(model.parameters(), lr=args.lr)   ## using ADAdelta as optimization algorithm

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma) ## Decays the learning rate of each parameter group by gamma every step_size epochs
    trainloss=[]
    accr=[]
    test_train_accr =[]
    test_train_loss=0
    for epoch in range(1, args.epochs + 1):        
        cn=0
        train(args, model, device, train_loader, optimizer, epoch, trainloss, test_train_accr, test_train_loss,args.epochs, args.batch_size,cn)
        test(model, device, test_loader,accr)
        np.savetxt("accuracy_ASK_RATQ_SNR_1e04.csv", accr, delimiter =", ", fmt ='% s')
        scheduler.step()    
    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")
    plt.plot(accr)
    
    plt.xlabel('epochs')
    plt.ylabel('Test accuracy')
    plt.show()
    plt.plot(trainloss)
    
    plt.xlabel('#batches')
    plt.ylabel('Train loss')
    plt.show()
    '''plt.plot(test_train_accr)
    plt.xlabel('Per 10 mini-batches')
    plt.ylabel('Training Accuracy')
    plt.show()'''

if __name__ == '__main__':
    main()

