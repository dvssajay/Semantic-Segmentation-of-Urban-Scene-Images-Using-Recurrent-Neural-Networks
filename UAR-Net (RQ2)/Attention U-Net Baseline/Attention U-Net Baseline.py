
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np
import sys, time, os, warnings 
from skimage.segmentation import mark_boundaries
import matplotlib.pylab as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


from torchvision.datasets import Cityscapes
import segmentation_models_pytorch as smp
from torchvision.transforms.functional import to_tensor, to_pil_image
from PIL import Image


# In[3]:


from albumentations import (HorizontalFlip,Compose,Resize,Normalize)

mean = [0.485, 0.456, 0.406] 
std = [0.229, 0.224, 0.225]
h,w=256,512

transform_train = Compose([ Resize(h,w),
                HorizontalFlip(p=0.5), 
                Normalize(mean=mean,std=std)])

transform_val = Compose( [ Resize(h,w),
                          Normalize(mean=mean,std=std)])


# In[4]:


class myCityscapes(Cityscapes):
    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert('RGB')

        targets = []
        for i, t in enumerate(self.target_type):
            if t == 'polygon':
                target = self._load_json(self.targets[index][i])
            else:
                target = Image.open(self.targets[index][i])

            targets.append(target)

        target = tuple(targets) if len(targets) > 1 else targets[0]

        if self.transforms is not None:
            sample= self.transforms(image=np.array(image), mask=np.array(target))
#            sample = self.transform(**sample)
            img = sample['image']
            target = sample['mask'] 
            #img, mask = self.transforms(np.array(image),np.array(target))
            
        img = to_tensor(img)
        mask = torch.from_numpy(target).type(torch.long)

        return img, mask


# In[5]:


train_ds = myCityscapes("./", split='train', mode='fine', target_type='semantic', transforms=transform_train, target_transform=None)
           #transforms=None)
test = myCityscapes("./", split='val', mode='fine', target_type='semantic', transforms=transform_val, target_transform=None)
#           transforms=None)


# In[6]:


train , val = torch.utils.data.random_split(train_ds, [2000,975])


# In[7]:


len(train)


# In[8]:


len(val)


# In[9]:


len(test)


# In[10]:


#number of classes presented in data
np.random.seed(0)
num_classes=35
COLORS = np.random.randint(0, 2, size=(num_classes+1, 3),dtype="uint8")


# In[11]:


def show_img_target(img, target):
    if torch.is_tensor(img):
        img=to_pil_image(img)
        target=target.numpy()
    for ll in range(num_classes):
        mask=(target==ll)
        img=mark_boundaries(np.array(img) , 
                            mask,
                            outline_color=COLORS[ll],
                            color=COLORS[ll])
    plt.imshow(img)


# In[12]:


def re_normalize (x, mean = mean, std= std):
    x_r= x.clone()
    for c, (mean_c, std_c) in enumerate(zip(mean, std)):
        x_r [c] *= std_c
        x_r [c] += mean_c
    return x_r


# In[13]:


#sample from training data
img, mask = train[3]
print(img.shape, img.type(),torch.max(img))
print(mask.shape, mask.type(),torch.max(mask))


# In[14]:


plt.figure(figsize=(20,20))

img_r= re_normalize(img)
plt.subplot(1, 3, 1) 
plt.imshow(to_pil_image(img_r))

plt.subplot(1, 3, 2) 
plt.imshow(mask)

plt.subplot(1, 3, 3) 
show_img_target(img_r, mask)


# In[15]:


#sample from validation data
img, mask = val[0]
print(img.shape, img.type(),torch.max(img))
print(mask.shape, mask.type(),torch.max(mask))


# In[16]:


plt.figure(figsize=(20,20))

img_r= re_normalize(img)
plt.subplot(1, 3, 1) 
plt.imshow(to_pil_image(img_r))

plt.subplot(1, 3, 2) 
plt.imshow(mask)

plt.subplot(1, 3, 3) 
show_img_target(img_r, mask)


# In[17]:


#sample from validation data
img, mask = test[0]
print(img.shape, img.type(),torch.max(img))
print(mask.shape, mask.type(),torch.max(mask))


# In[18]:


plt.figure(figsize=(20,20))

img_r= re_normalize(img)
plt.subplot(1, 3, 1) 
plt.imshow(to_pil_image(img_r))

plt.subplot(1, 3, 2) 
plt.imshow(mask)

plt.subplot(1, 3, 3) 
show_img_target(img_r, mask)


# In[19]:


#defining Dataloaders
from torch.utils.data import DataLoader
train_dl = DataLoader(train, batch_size=3, shuffle=True)
val_dl = DataLoader(val, batch_size=3, shuffle=False)
test_dl =DataLoader(test, batch_size=1, shuffle=False)


# In[20]:


class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi


# In[21]:


class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )


    def forward(self,x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x
    
class single_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(single_conv,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.conv(x)
        return x


# In[22]:


class AttU_Net(nn.Module):
    def __init__(self,img_ch=3,output_ch=35):
        super(AttU_Net,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=64)
        self.Conv2 = conv_block(ch_in=64,ch_out=128)
        self.Conv3 = conv_block(ch_in=128,ch_out=256)
        self.Conv4 = conv_block(ch_in=256,ch_out=512)
        self.Conv5 = conv_block(ch_in=512,ch_out=1024)

        self.Up5 = up_conv(ch_in=1024,ch_out=512)
        self.Att5 = Attention_block(F_g=512,F_l=512,F_int=256)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512,ch_out=256)
        self.Att4 = Attention_block(F_g=256,F_l=256,F_int=128)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)
        
        self.Up3 = up_conv(ch_in=256,ch_out=128)
        self.Att3 = Attention_block(F_g=128,F_l=128,F_int=64)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)
        
        self.Up2 = up_conv(ch_in=128,ch_out=64)
        self.Att2 = Attention_block(F_g=64,F_l=64,F_int=32)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)


    def forward(self,x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(d5,x4)
        d5 = torch.cat((x4,d5),dim=1)        
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        x3 = self.Att4(d4,x3)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(d3,x2)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(d2,x1)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1


# In[23]:


model = AttU_Net()


# In[24]:


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model=model.to(device)


# In[25]:


criterion = nn.CrossEntropyLoss(reduction="sum")
from torch import optim
opt = optim.Adam(model.parameters(), lr=1e-6)


# In[26]:


def loss_batch(loss_func, output, target, opt=None):   
    loss = loss_func(output, target)
    
    if opt is not None:
        opt.zero_grad()
        loss.backward()
        opt.step()

    return loss.item(), None


# In[27]:


from torch.optim.lr_scheduler import ReduceLROnPlateau
lr_scheduler = ReduceLROnPlateau(opt, mode='min',factor=0.5, patience=20,verbose=1)


# In[28]:


def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group['lr']

current_lr=get_lr(opt)
print('current lr={}'.format(current_lr))


# In[29]:


def loss_epoch(model,loss_func,dataset_dl,sanity_check=False,opt=None):
    running_loss=0.0
    len_data=len(dataset_dl.dataset)
    for xb, yb in dataset_dl:
        xb=xb.to(device)
        yb=yb.to(device)
        output=model(xb)
        loss_b, metric_b = loss_batch(loss_func, output, yb, opt)
        running_loss += loss_b
        if sanity_check is True:
            break
    loss=running_loss/float(len_data)
    return loss, None


# In[30]:


import copy
def train_val(model, params):
    num_epochs=params["num_epochs"]
    loss_func=params["loss_func"]
    opt=params["optimizer"]
    train_dl=params["train_dl"]
    val_dl=params["val_dl"]
    sanity_check=params["sanity_check"]
    lr_scheduler=params["lr_scheduler"]
    path2weights=params["path2weights"]

    loss_history={
        "train": [],
        "val": []}
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss=float('inf')

    for epoch in range(num_epochs):
        current_lr=get_lr(opt)
        print('Epoch {}/{}, current lr={}'.format(epoch, num_epochs - 1, current_lr))   

        model.train()
        train_loss, _ = loss_epoch(model,loss_func,train_dl,sanity_check,opt)
        loss_history["train"].append(train_loss)
        
        model.eval()
        with torch.no_grad():
            val_loss, _ = loss_epoch(model,loss_func,val_dl,sanity_check)
        loss_history["val"].append(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), path2weights)
            print("Copied best model weights!")
            
        lr_scheduler.step(val_loss)
        if current_lr != get_lr(opt):
            print("Loading best model weights!")
            model.load_state_dict(best_model_wts)

        print("train loss: %.6f" %(train_loss))
        print("val loss: %.6f" %(val_loss))
        print("-"*10) 
    model.load_state_dict(best_model_wts)
    return model, loss_history


# In[31]:


start = time.time()

import os
path2models= "./ARnet_Experiments/Unet_Baseline2"
if not os.path.exists(path2models):
        os.mkdir(path2models)
params_train={
    "num_epochs": 150,
    "optimizer": opt,
    "loss_func": criterion,
    "train_dl": train_dl,
    "val_dl": val_dl,
    "sanity_check": False,
    "lr_scheduler": lr_scheduler,
    "path2weights": path2models+"weights.pt",}
model,loss_hist=train_val(model,params_train)

end = time.time()
print("TIME TOOK {:3.2f}MIN".format((end - start )/60))


# In[32]:


torch.save(model.state_dict(), "./ARnet_Experiments/Unet_Baseline2M")


# In[33]:


num_epochs=params_train["num_epochs"]
plt.figure(figsize=(30,30))
plt.title("Train-Val Loss")
plt.plot(range(1,num_epochs+1),loss_hist["train"],label="train")
plt.plot(range(1,num_epochs+1),loss_hist["val"],label="val")
plt.ylabel("Loss")
plt.xlabel("Training Epochs")
plt.legend()
plt.show()


# In[31]:


model.load_state_dict(torch.load("./ARnet_Experiments/Unet_Baseline2weights.pt"))
model.eval()


# In[41]:


from torch.autograd import Variable


# In[42]:


start = time.time()
test_loss, _ = loss_epoch(model,criterion,test_dl,opt)
end = time.time()
print("TIME TOOK {:3.2f}MIN".format((end - start )/60))


# In[43]:


print(test_loss)


# In[32]:


out_dl =DataLoader(test, batch_size=5, shuffle=False)


# In[33]:


len_data=len(out_dl.dataset)


# In[34]:


print(len_data)


# In[35]:


for xb, yb in out_dl:
    xb=xb.to(device)
    yb=yb.to(device)
    output=model(xb)
    break


# In[36]:


pred = torch.argmax(output, dim=1)


# In[37]:


mapc = np.array([[0.        , 0.        , 0.        ],
       [0.        , 0.        , 0.        ],
       [0.        , 0.        , 0.        ],
       [0.        , 0.        , 0.        ],
       [0.07843137, 0.07843137, 0.07843137],
       [0.43529412, 0.29019608, 0.        ],
       [0.31764706, 0.        , 0.31764706],
       [0.50196078, 0.25098039, 0.50196078],
       [0.95686275, 0.1372549 , 0.90980392],
       [0.98039216, 0.66666667, 0.62745098],
       [0.90196078, 0.58823529, 0.54901961],
       [0.2745098 , 0.2745098 , 0.2745098 ],
       [0.4       , 0.4       , 0.61176471],
       [0.74509804, 0.6       , 0.6       ],
       [0.70588235, 0.64705882, 0.70588235],
       [0.58823529, 0.39215686, 0.39215686],
       [0.58823529, 0.47058824, 0.35294118],
       [0.6       , 0.6       , 0.6       ],
       [0.6       , 0.6       , 0.6       ],
       [0.98039216, 0.66666667, 0.11764706],
       [0.8627451 , 0.8627451 , 0.        ],
       [0.41960784, 0.55686275, 0.1372549 ],
       [0.59607843, 0.98431373, 0.59607843],
       [0.2745098 , 0.50980392, 0.70588235],
       [0.8627451 , 0.07843137, 0.23529412],
       [1.        , 0.        , 0.        ],
       [0.        , 0.        , 0.55686275],
       [0.        , 0.        , 0.2745098 ],
       [0.        , 0.23529412, 0.39215686],
       [0.        , 0.        , 0.35294118],
       [0.        , 0.        , 0.43137255],
       [0.        , 0.31372549, 0.39215686],
       [0.        , 0.        , 0.90196078],
       [0.46666667, 0.04313725, 0.1254902 ],
       [0.        , 0.        , 0.55686275]])


# In[38]:


import numpy as np


# In[39]:


def color_img(mask):
    out = np.empty([256,512,3])
    for i in range(256):
        for j in range(512):
            x = (mask[i][j]).item()
            out[i][j] = mapc[x]
    return out


# In[40]:


plt.imshow(color_img(pred[0]))


# In[49]:


plt.figure(figsize=(30,30))
    
img_r= re_normalize(xb[0].cpu())
plt.subplot(1, 3, 1) 
plt.imshow(to_pil_image(img_r))
    
plt.subplot(1, 3, 2) 
plt.imshow(yb[0].cpu())
    
plt.subplot(1, 3, 3) 
plt.imshow(pred[0].cpu())


# In[41]:


plt.imshow(color_img(pred[0].cpu()))
plt.savefig('AttUnet1.png', dpi = 300)


# In[50]:


plt.figure(figsize=(30,30))
    
img_r= re_normalize(xb[1].cpu())
plt.subplot(1, 3, 1) 
plt.imshow(to_pil_image(img_r))
    
plt.subplot(1, 3, 2) 
plt.imshow(yb[1].cpu())
    
plt.subplot(1, 3, 3) 
plt.imshow(pred[1].cpu())


# In[42]:


plt.imshow(color_img(pred[1].cpu()))
plt.savefig('AttUnet2.png', dpi = 300)


# In[51]:


plt.figure(figsize=(30,30))
    
img_r= re_normalize(xb[2].cpu())
plt.subplot(1, 3, 1) 
plt.imshow(to_pil_image(img_r))
    
plt.subplot(1, 3, 2) 
plt.imshow(yb[2].cpu())
    
plt.subplot(1, 3, 3) 
plt.imshow(pred[2].cpu())


# In[43]:


plt.imshow(color_img(pred[2].cpu()))
plt.savefig('AttUnet3.png', dpi = 300)


# In[52]:


plt.figure(figsize=(30,30))
    
img_r= re_normalize(xb[3].cpu())
plt.subplot(1, 3, 1) 
plt.imshow(to_pil_image(img_r))
    
plt.subplot(1, 3, 2) 
plt.imshow(yb[3].cpu())
    
plt.subplot(1, 3, 3) 
plt.imshow(pred[3].cpu())


# In[44]:


plt.imshow(color_img(pred[3].cpu()))
plt.savefig('AttUnet4.png', dpi = 300)


# In[53]:


plt.figure(figsize=(30,30))
    
img_r= re_normalize(xb[4].cpu())
plt.subplot(1, 3, 1) 
plt.imshow(to_pil_image(img_r))
    
plt.subplot(1, 3, 2) 
plt.imshow(yb[4].cpu())
    
plt.subplot(1, 3, 3) 
plt.imshow(pred[4].cpu())


# In[45]:


plt.imshow(color_img(pred[4].cpu()))
plt.savefig('AttUnet5.png', dpi = 300)


# In[56]:


o_dl =DataLoader(test, batch_size=1, shuffle=False)


# In[57]:


SMOOTH = 1e-6

def iou_pytorch(outputs, labels):
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    #outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
    
    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))         # Will be zzero if both are 0
    
    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0
    
    thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds
    
    return thresholded,iou  # Or thresholded.mean() if you are interested in average across the batch


# In[58]:


import pandas as pd


# In[59]:


file = pd.read_csv("./Results_B1.csv")


# In[60]:


file.head()


# In[61]:


pix = pd.read_csv("./Res-pic.csv")


# In[62]:


pix.head()


# In[63]:


iou_sum = torch.zeros(1)
t_sum = torch.zeros(1)
acc = torch.zeros(1)
counter = 0
for xb, yb in o_dl:
    xb=xb.to(device)
    yb=yb.to(device)
    output=model(xb)
    pixel = criterion(output,yb)
    pred = torch.argmax(output, dim=1)
    t,i = iou_pytorch(pred,yb)
    print(i)
    print(pixel)
    pix['U-Net IOU'][counter]= int(pixel.item())
    file['U-Net IOU'][counter]= round(i.item()*100,2)
    counter += 1
    iou_sum += i.sum()
    


# In[64]:


file.to_csv("./ui1.csv")


# In[65]:


pix.to_csv("./ua1.csv")


# In[66]:


a = pd.read_csv("./ua1.csv")


# In[67]:


a.head()


# In[68]:


b = pd.read_csv("./ui1.csv")


# In[69]:


b.head()


# In[35]:


iou_sum = torch.zeros(1)
t_sum = torch.zeros(1)
for xb, yb in o_dl:
    xb=xb.to(device)
    yb=yb.to(device)
    output=model(xb)
    pred = torch.argmax(output, dim=1)
    t,i = iou_pytorch(pred,yb)
    iou_sum += i.sum()
    t_sum += t.sum()
    
    


# In[36]:


print(iou_sum/500)


# In[37]:


print(t_sum/500)


# In[38]:


from torchsummary import summary


# In[39]:


summary(model, (3, 256, 512))

