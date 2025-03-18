import torch.nn as nn
import torch


    
class mask2(nn.Module):
    def __init__(self,):
        super(mask2, self).__init__()
  
        self.generation = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True), 
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            #nn.Conv2d(128, 512, kernel_size=1)
            )

    def forward(self, x):
        #x=self.align(x)
        N, C, H, W = x.shape

        device = x.device
        #mat = torch.rand((N,C,1,1)).to(device)
        mat = torch.rand((N,1,H,W)).to(device)
        mat = torch.where(mat < 0.15, 0, 1).to(device)

        new_fea = torch.mul(x, mat)
        new_fea = self.generation(new_fea)
       
        
        return new_fea
        
class mask3(nn.Module):
    def __init__(self,):
        super(mask3, self).__init__()
       
        self.generation = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True), 
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            #nn.Conv2d(256, 512, kernel_size=1)
            )
            
        

    def forward(self, x):
        #x=self.align(x)
        N, C, H, W = x.shape

        device = x.device
        #mat = torch.rand((N,C,1,1)).to(device)
        mat = torch.rand((N,1,H,W)).to(device)
        mat = torch.where(mat < 0.15, 0, 1).to(device)

        new_fea = torch.mul(x, mat)
      
        new_fea = self.generation(new_fea)
        
        
        
        return new_fea
        
