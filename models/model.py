import torch
import pytorch_lightning as pl
from torch.optim.lr_scheduler import MultiplicativeLR
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
from sklearn.metrics import classification_report
from torch.nn import functional as F
from torch import nn
from scikitplot.metrics import plot_confusion_matrix
from efficientnet_pytorch import EfficientNet
from torchvision import models, transforms
import timm    

class LesionClassification(pl.LightningModule):

    def __init__(self, model, num_classes, lr=1e-4, lr_decay=0.9, mode="train", checkpoint=None):
        super().__init__()
        self.model = timm.create_model(
            model, pretrained=True, num_classes=num_classes
        )
        self.lr = lr
        self.lr_decay = lr_decay
        self.mode = mode
        self.model_name = mode

    def forward(self, x):
        if self.mode == "train":
            x = x['image']
        out = self.model(x)
        return out
    
    def configure_optimizers(self):        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        lambd = lambda epoch: self.lr_decay
        scheduler = MultiplicativeLR(self.optimizer, lr_lambda = lambd)
        return [self.optimizer], [scheduler]
    
    def training_step(self, batch, batch_idx):
        x,y =  batch
        y_pred = self(x)
        loss = F.cross_entropy(y_pred,y)
        self.log("train_loss",loss,prog_bar=True,logger=True)

        return {'loss':loss}

    def validation_step(self,batch,batch_idx):
        x,y = batch
        y_pred = self(x)
        loss = F.cross_entropy(y_pred.to(self.device),
                            y.to(self.device)).cuda()
       
        self.log("val_loss",loss,prog_bar=True,logger=True)
        return {'loss':loss, 'y':y, 'y_pred':y_pred}
    
    def validation_epoch_end(self, outputs):
        scores = []
        for i, out in enumerate(outputs):
            y, y_pred = out['y'], out['y_pred']
            y = y.reshape(-1, 1)
            if i == 0:
                all_y = y
                all_ypred = y_pred
            else:
                all_y = torch.cat((all_y,y),0)
                all_ypred = torch.cat((all_ypred,y_pred),0)
            

        all_ypred = torch.argmax(all_ypred, dim=1).cpu().detach().numpy()
        all_y = all_y.cpu().detach().numpy()
        self.logger.experiment.log_text("classification_report",str(classification_report(all_y,all_ypred)))
        fig, ax = plt.subplots(figsize=(4,4))
        plot_confusion_matrix(all_y, all_ypred, ax=ax)
        self.logger.experiment.log_image('confusion_matrix', fig)
        self.optimizer.step()
        
