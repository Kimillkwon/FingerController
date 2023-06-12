import time
import torch
from tqdm import tqdm
import copy

class Trainer:
    def __init__(self, model, params):
        self.model = model
        self.num_epochs=params["num_epochs"]
        self.loss_func=params["loss_func"]
        self.opt=params["optimizer"]
        self.train_loader=params["train_loader"]
        self.validation_loader=params["validation_loader"]
        self.lr_scheduler=params["lr_scheduler"]
        self.device = params["device"]
        self.save_path = params["save_path"]

    def train_val(self):
        loss_history = {'train': [], 'val': []}
        metric_history = {'train': [], 'val': []}
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_loss = float('inf')
        
        for epoch in range(self.num_epochs):
            start_time = time.time()
            current_lr = self._get_lr()
            print('Epoch {}/{}, current lr={}'.format(epoch, self.num_epochs - 1, current_lr))
            
            self.model.train()
            train_loss, train_metric = self._loss_epoch(True)
            loss_history['train'].append(train_loss)
            metric_history['train'].append(train_metric)

            self.model.eval()
            with torch.no_grad():
                val_loss, val_metric = self._loss_epoch(False)

            if val_loss < best_loss:
                best_loss = val_loss
                best_model_wts = copy.deepcopy(self.model.state_dict())

                torch.save(self.model.state_dict(), self.save_path)

            loss_history['val'].append(val_loss)
            metric_history['val'].append(val_metric)

            self.lr_scheduler.step(val_loss)

            print('train loss: %.8f, train accuracy: %.2f, val loss: %.8f, val accuracy: %.2f, time: %.4fs' %(train_loss, train_metric, val_loss, val_metric, (time.time()-start_time)))
            print('-'*10)
        
        self.model.load_state_dict(best_model_wts)
        return self.model, loss_history, metric_history

    def _metric_batch(self, output, target):
        _, predicted = torch.max(output.data, 1)
        _, target = torch.max(target, 1)
        corrects = (predicted == target).sum().item()
        return corrects

    def _get_lr(self):
        for param_group in self.opt.param_groups:
            return param_group['lr']

    def _loss_batch(self, outputs, target, trainFlag):
        loss = self.loss_func(outputs, target)
        metric_b = self._metric_batch(outputs, target)
        
        if trainFlag:
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
        
        return loss.item(), metric_b

    def _loss_epoch(self, trainFlag=True):
        running_loss = 0.0
        running_metric = 0.0
        if trainFlag:
            loader = self.train_loader
        else:
            loader = self.validation_loader

        total = 0
        
        for images, labels in tqdm(loader):
            images, labels = images.to(self.device), labels.to(self.device)
            outputs = self.model(images)
            loss_b, metric_b = self._loss_batch(outputs, labels, trainFlag)
            running_loss += loss_b
            
            if metric_b is not None:
                running_metric += metric_b
            total += 1

        loss = running_loss / total
        metric = running_metric / total
        return loss, metric
