import torch
import torch.optim as opt
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from PIL import Image
import numpy as np
import pathlib

def imshow(imgs, epoch):
    imgs = torchvision.utils.make_grid(imgs, nrow=4)
    npimgs = imgs.numpy()
    plt.figure(figsize=(8,8))
    plt.imshow(np.transpose(npimgs, (1,2,0)), cmap='gray')
    plt.xticks([])
    plt.yticks([])
    name = "results/" +str(epoch) + ".png"
    plt.savefig('results/foo.png')
    #plt.show()

def load_data(root_folder):
    image_size = (64, 64)
    dataset = torchvision.datasets.ImageFolder(root=root_folder,
                                               transform = transforms.Compose([transforms.Resize(image_size),
                                                transforms.CenterCrop(image_size),
                                                transforms.ToTensor()]))
    cars_data = []
    for i in range(len(dataset.targets)):
        if dataset.targets[i] == 1:
            cars_data.append(i)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, 
                                                sampler = torch.utils.data.sampler.SubsetRandomSampler(cars_data))
    return train_loader

source_folder = "/home/swapnil/Desktop/IE643_Assignment_4/natural_images/"
train_loader = load_data(source_folder)

# data = iter(train_loader)
# img , labels = data.next()
# print(img.shape, labels.shape)


'''
# define the dataset loader [Custom dataloader]
class Dataset:
	def __init__(self, root=None, transform=None):
		"""Dataset for natural images data
		Args:
			root: the root of natural images folder
		"""
		self.root = pathlib.Path(root)
		self.transform = transform
		self.seqs = np.sort(os.listdir(self.root))
		self.image_data = []
		self.label_data = []
		for i,seq in enumerate(self.seqs):
			seq_path = os.path.join(self.root,seq)
			image_list = np.sort(os.listdir(seq_path))
			for image_file in image_list:
				self.image_data.append(os.path.join(seq_path,image_file))
				self.label_data.append(i)

	def __getitem__(self, index):
		image_file = self.image_data[index]
		image = cv2.imread(str(image_file))
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		image = Image.fromarray(image)
		image = self.transform(image)
		label = self.label_data[index]
		return image, label

	def __len__(self):
		return len(self.label_data)
'''

nc = 3    # Number of channels in the training images. For color images this is 3
nz = 100  # Size of z latent vector (i.e. size of generator input)
ngf = 64  # Size of feature maps in generator
ndf = 64  # Size of feature maps in discriminator
# Batch_size = 32

#Discriminator Code
class DC_Dis(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            # input is 3 x 64 x 64
            nn.Conv2d(nc, ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.model(input)

# Generator Code
class DC_Gen(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.model(input)


G = DC_Gen()
D = DC_Dis()

lr = 1e-3  #learning rate
#optimizers for both models
g_opt = opt.Adam(G.parameters(), lr=lr)
d_opt = opt.Adam(D.parameters(), lr=lr)


G_loss_epoch = []
D_loss_epoch = []
total_epochs = 100
for epoch in range(total_epochs):
    G_loss_run = 0.0
    D_loss_run = 0.0
    
    for i, data in enumerate(train_loader):
        X, _ = data
        #X = X.view(X.size(0), -1)
        mb_size = X.size(0)
        
        one_labels = torch.ones(mb_size, 1, 1, 1)
        zero_labels = torch.zeros(mb_size, 1, 1, 1)
        
        z = torch.randn(mb_size, nz, 1 , 1)
        

        D_real = D(X)
        D_fake = D(G(z))

        D_real_loss = F.binary_cross_entropy(D_real, one_labels)  #loss -(1/m)(log D(x))
        D_fake_loss = F.binary_cross_entropy(D_fake, zero_labels)  #loss -(1/m)(log(1-D(G(z))))
        D_loss = D_real_loss + D_fake_loss
        
        d_opt.zero_grad()
        D_loss.backward()
        d_opt.step()
        
        z = torch.randn(mb_size, nz, 1, 1)
        
        D_fake = D(G(z))
        G_loss = F.binary_cross_entropy(D_fake, one_labels)  #loss -(1/m)(log (1-D(G(z))))
        
        g_opt.zero_grad()
        G_loss.backward()
        g_opt.step()
        
        G_loss_run += G_loss.item()
        D_loss_run += D_loss.item()

        #print("Batch: %d" %i)
        
    print('Epoch:{},   G_loss:{},    D_loss:{}'.format(epoch, G_loss_run/(i+1), D_loss_run/(i+1)))
    G_loss_epoch.append(G_loss_run/(i+1))
    D_loss_epoch.append(D_loss_run/(i+1))
    
    

    if(epoch%10 == 0):
        # priting a sample of 16 images by gan
        samples = G(torch.randn(16, nz, 1, 1)).detach()
        samples = samples.view(samples.size(0), 3, 64, 64)
        imshow(samples, epoch)

# Printing plots of Generator and Discriminiator error with epochs
plt.plot(range(1,total_epochs+1), G_loss_epoch, D_loss_epoch)
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.title("Losses vs Training Epochs")
plt.legend(("Generator Loss", "Discriminiator Loss"))
plt.savefig("Loss_vs_Epochs.png")