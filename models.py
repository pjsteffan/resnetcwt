import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import math
import lightning as L


from torchvision.models import resnet18

class Encoder(L.LightningModule):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_layers: int):
        super(Encoder, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size,output_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.shape[0],self.hidden_size).to(x.device).float()
        
        # Forward propagate through the GRU
        out, hidden = self.gru(x, h0)


        out = self.fc(out)
  
        
        # Return the hidden state of the last layer
        return out, hidden
    

class TrainerBase(L.LightningModule):
    def __init__(self):
        super(TrainerBase,self).__init__()

    
    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        data, labels = batch
        
        
        #data = data.unsqueeze(-1)  # Add a channel dimension
        data = data.to(self.device).float()
        labels = labels.to(self.device).float()
        outputs = self(data)

        loss = self.criterion(outputs.squeeze(0), labels.type(torch.int64))

        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        data, labels = batch
  
        
        #data = data.unsqueeze(-1)  # Add a channel dimension
        data = data.to(self.device).float()
        labels = labels.to(self.device).float()
        outputs = self(data)

        loss = self.criterion(outputs.squeeze(0), labels.type(torch.int64))
        
        
        outputs_numpy = outputs.squeeze(0).cpu().detach().numpy()
        outputs_numpy = outputs_numpy.argmax(1)

        labels_numpy = labels.type(torch.int64).cpu().detach().numpy()
        
        
        # Calculate various metrics
        accuracy = accuracy_score(labels_numpy,outputs_numpy)
        precision = precision_score(labels_numpy,outputs_numpy, average='weighted')
        recall = recall_score(labels_numpy,outputs_numpy, average='weighted')
        f1 = f1_score(labels_numpy,outputs_numpy, average='weighted')
        conf_matrix = confusion_matrix(labels_numpy,outputs_numpy)
        self.log('accuracy', accuracy)
        self.log('precision', precision)
        self.log('recall',recall)
        self.log('F1 Score', f1)
        print(f"Confusion Matrix\n{conf_matrix}")
        
        
        
        self.log('train_loss', loss)
        return loss

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        data, labels = batch
        

        
        data = data.unsqueeze(-1)  # Add a channel dimension
        data = data.to(self.device).float()
        labels = labels.to(self.device).float()
        outputs = self(data)

        loss = self.criterion(outputs.squeeze(0), labels.type(torch.int64))
        self.log('train_loss', loss)
        return loss
    
    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=1e-3)

   

class GRUClassifier(TrainerBase):
    def __init__(self, input_size: int, num_layers: int, hidden_size: int, output_size: int, out_size1: int, out_size2: int, logit_size: int,weight=None):
        super(GRUClassifier, self).__init__()
        self.save_hyperparameters()
        self.encoder = Encoder(input_size, hidden_size, output_size, num_layers)
        self.fcout1 = nn.Linear(hidden_size, out_size1)
        self.fcout2 = nn.Linear(out_size1, out_size2)
        self.fcout3 = nn.Linear(out_size2, logit_size)
        self.criterion = nn.CrossEntropyLoss()#Removed Weight


    def forward(self, source: torch.Tensor) -> torch.Tensor:
        # Encode the source sequence
        out, hidden = self.encoder(source)

        
        fc_out1 = self.fcout1(hidden.squeeze(0))
        fc_out2 = self.fcout2(fc_out1)
        outputs = self.fcout3(fc_out2)

        return outputs

class ResNet18(TrainerBase):
    def __init__(self, num_classes):
        super(ResNet18, self).__init__()
        self.model = resnet18(num_classes=num_classes)
        self.criterion = nn.CrossEntropyLoss()
    def forward(self,source):
        out = self.model(source)
        return out



def equal_var_init(model):
    for name, param in model.named_parameters():
        if name.endswith(".bias"):
            param.data.fill_(0)
        elif '.gru.bias' in name:
            param.data.fill_(0)
        else:
            param.data.normal_(std=1.0 / math.sqrt(param.shape[1]))