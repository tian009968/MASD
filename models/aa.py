import torch.nn as nn
import torch
'''
class SPP111(nn.Module):
    def __init__(self,):
        super(SPP111, self).__init__()
       
        self.generation = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True), 
            nn.Conv2d(64, 64, kernel_size=3, padding=1))
        

    def forward(self, x):
        
        N, C, H, W = x.shape
        device = x.device

       
        mask_ratio = 0.15
        patch_size = 2  
        
        num_patches_h = H // patch_size
        num_patches_w = W // patch_size
        total_patches = num_patches_h * num_patches_w
        num_mask_patches = int(total_patches * mask_ratio)

       
        mat = torch.ones((N, 1, H, W), dtype=torch.float).to(device)

        for i in range(N):
           
            mask_indices = torch.randperm(total_patches)[:num_mask_patches]
            for idx in mask_indices:
                patch_h = (idx // num_patches_w) * patch_size
                patch_w = (idx % num_patches_w) * patch_size
                mat[i, :, patch_h:patch_h + patch_size, patch_w:patch_w + patch_size] = 0

       
        masked_fea = x * mat
        new_fea = self.generation(masked_fea)

        return new_fea
        
        
        
class SPP222(nn.Module):
    def __init__(self,):
        super(SPP222, self).__init__()
  
        self.generation = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True), 
            nn.Conv2d(128, 128, kernel_size=3, padding=1))
        

    def forward(self, x):
        N, C, H, W = x.shape
        device = x.device

       
        mask_ratio = 0.15
        patch_size = 2  
        
        num_patches_h = H // patch_size
        num_patches_w = W // patch_size
        total_patches = num_patches_h * num_patches_w
        num_mask_patches = int(total_patches * mask_ratio)

       
        mat = torch.ones((N, 1, H, W), dtype=torch.float).to(device)

        for i in range(N):
           
            mask_indices = torch.randperm(total_patches)[:num_mask_patches]
            for idx in mask_indices:
                patch_h = (idx // num_patches_w) * patch_size
                patch_w = (idx % num_patches_w) * patch_size
                mat[i, :, patch_h:patch_h + patch_size, patch_w:patch_w + patch_size] = 0

       
        masked_fea = x * mat
        new_fea = self.generation(masked_fea)

        return new_fea
        
class SPP333(nn.Module):
    def __init__(self,):
        super(SPP333, self).__init__()
       
        self.generation = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True), 
            nn.Conv2d(256, 256, kernel_size=3, padding=1))
        

    def forward(self, x):
        N, C, H, W = x.shape
        device = x.device

       
        mask_ratio = 0.15
        patch_size = 2  
        
        num_patches_h = H // patch_size
        num_patches_w = W // patch_size
        total_patches = num_patches_h * num_patches_w
        num_mask_patches = int(total_patches * mask_ratio)

       
        mat = torch.ones((N, 1, H, W), dtype=torch.float).to(device)

        for i in range(N):
           
            mask_indices = torch.randperm(total_patches)[:num_mask_patches]
            for idx in mask_indices:
                patch_h = (idx // num_patches_w) * patch_size
                patch_w = (idx % num_patches_w) * patch_size
                mat[i, :, patch_h:patch_h + patch_size, patch_w:patch_w + patch_size] = 0

       
        masked_fea = x * mat
        new_fea = self.generation(masked_fea)

        return new_fea





'''
'''
class SPP111(nn.Module):
    def __init__(self,):
        super(SPP111, self).__init__()
       
        self.generation = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True), 
            nn.Conv2d(64, 64, kernel_size=3, padding=1))
        

    def forward(self, x):
        #x=self.align(x)
        N, C, H, W = x.shape

        device = x.device
        #mat = torch.rand((N,C,1,1)).to(device)
        mat = torch.rand((N,1,H,W)).to(device)
        mat = torch.where(mat < 0.15, 0, 1).to(device)

        masked_fea = torch.mul(x, mat)
        new_fea = self.generation(masked_fea)
        
        
        return new_fea
        
class SPP222(nn.Module):
    def __init__(self,):
        super(SPP222, self).__init__()
  
        self.generation = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True), 
            nn.Conv2d(128, 128, kernel_size=3, padding=1))
        

    def forward(self, x):
        #x=self.align(x)
        N, C, H, W = x.shape

        device = x.device
        #mat = torch.rand((N,C,1,1)).to(device)
        mat = torch.rand((N,1,H,W)).to(device)
        mat = torch.where(mat < 0.15, 0, 1).to(device)

        masked_fea = torch.mul(x, mat)
        new_fea = self.generation(masked_fea)
        
        return new_fea
        
class SPP333(nn.Module):
    def __init__(self,):
        super(SPP333, self).__init__()
       
        self.generation = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True), 
            nn.Conv2d(256, 256, kernel_size=3, padding=1))
        

    def forward(self, x):
        #x=self.align(x)
        N, C, H, W = x.shape

        device = x.device
        #mat = torch.rand((N,C,1,1)).to(device)
        mat = torch.rand((N,1,H,W)).to(device)
        mat = torch.where(mat < 0.15, 0, 1).to(device)

        masked_fea = torch.mul(x, mat)
        new_fea = self.generation(masked_fea)
        
        
        return new_fea
'''
import torch.nn as nn
import torch


    
class SPP222(nn.Module):
    def __init__(self,):
        super(SPP222, self).__init__()
  
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
        
class SPP333(nn.Module):
    def __init__(self,):
        super(SPP333, self).__init__()
       
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
        
