import torch
import torch.nn as nn

class GF_IF(nn.Module):
    def __init__(self):
        super(GF_IF, self).__init__()
        
        self.fc_tensor1 = nn.Sequential(
            nn.Linear(768, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512)
        )
        
        self.fc_tensor2 = nn.Sequential(
            nn.Linear(512 + 19, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512)
        )
        
        self.fc_tensor3 = nn.Sequential(
            nn.Linear(512 + 26, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512)
        )
        
        self.fc_tensor4 = nn.Sequential(
            nn.Linear(512 + 10, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512)
        )
        
        self.fc_tensor5 = nn.Sequential(
            nn.Linear(512 + 19, 1024), 
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2)
        )

    def forward(self, x1, x2, x3, x4, x5):
        x1 = self.fc_tensor1(x1)
        
        x2 = torch.cat((x1, x2), dim=1)
        x2 = self.fc_tensor2(x2)
        
        x3 = torch.cat((x2, x3), dim=1)
        x3 = self.fc_tensor3(x3)
        
        x4 = torch.cat((x3, x4), dim=1)
        x4 = self.fc_tensor4(x4)
        
        x5 = torch.cat((x4, x5), dim=1)
        x5 = self.fc_tensor5(x5)
        
        return x5 # output