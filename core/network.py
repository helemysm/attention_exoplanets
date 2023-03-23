import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import math
import os
import time
import tqdm

from random import gauss

from model import ScheduledOptimizer, model_clf
from box import box_from_file

import logging

logger = logging.getLogger("exoclf")

 
    
class Network:

    '''
    Main class to define the parameters and training of the model
    
    parameters
    model : model_clf implemented in the model class
    criterion: loss function, for example nn.CrossEntropyLoss()
    optimizer: class for learning rate scheduling (ScheduledOptimizer)
    experiment: unit of measurable experiments that defines a 
                single run. https://www.comet.com/docs/v2/api-and-sdk/python-sdk/reference/Experiment/

    '''
    
    def __init__(self, model, criterion, optimizer, experiment) -> None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            self.model = model.to(self.device)

            self.optimizer = opt
            self.criterion = criterion
            self.experiment = experiment

            self._start_epoch = 0
            self.__num_classes = None
            self._is_parallel = False

            if torch.cuda.device_count() > 1:
                self.model = nn.DataParallel(self.model)
                self._is_parallel = True

                notice = "Running on {} GPUs.".format(torch.cuda.device_count())
                print("\033[33m" + notice + "\033[0m")

    
    def fit(self, loader: Dict[str, DataLoader], epochs: int, checkpoint_path: str = None, validation: bool = True) -> None:
     '''
     Method to the model train given parameters and datasets
     
     parameters
     loader: DataLoader 
     epochs: number of epochs to train the model
     checkpoint_path: model previously trained
     
     '''

        len_of_train_dataset = len(loader["train"].dataset)
        epochs = epochs + self._start_epoch

        if validation:
            len_of_val_dataset = len(loader["val"].dataset)
            
        steps_iter = 0
        steps_iter_val = 0
        acc_best = 0

        for epoch in range(self._start_epoch, epochs):

            print('epoch ', (epoch % 100 == 0))
           
            if checkpoint_path is not None and epoch % 100 == 0:
                self.save_model(checkpoint_path)
            with self.experiment.train():
                correct = 0.0
                total = 0.0

                self.model.train()
                pbar = tqdm(total=len_of_train_dataset)

                  
                for x, x_cen, x_global, x_cen_global, x_param, x_timel, x_timeg, y in loader["train"]:
                    b_size = y.shape[0]
                    total += y.shape[0]

                    x = x.to(self.device) if isinstance(x, torch.Tensor) else [i.to('cpu') for i in x]
                    x_cen = x_cen.to(self.device) if isinstance(x_cen, torch.Tensor) else [i.to(self.device) for i in x_cen]
                    x_param = x_param.to(self.device) if isinstance(x_param, torch.Tensor) else [i.to(self.device) for i in x_param]
                    
                    x_global = x_global.to(self.device) if isinstance(x_global, torch.Tensor) else [i.to(self.device) for i in x_global]
                    x_cen_global = x_cen_global.to(self.device) if isinstance(x_cen_global, torch.Tensor) else [i.to(self.device) for i in x_cen_global]

                    y = y.to(self.device)

                    pbar.set_description(
                        "\033[36m" + "Training" + "\033[0m" + " - Epochs: {:03d}/{:03d}".format(epoch+1, epochs)
                    )
                    pbar.update(b_size)

                    self.optimizer.zero_grad()
                    
                    for batch_idx in range(x.shape[0]):
                        
                        value = x[batch_idx].mean()
                        value_cen = x_cen[batch_idx].mean()
                        noise = np.random.uniform(0, value)
                        noise_cen = np.random.uniform(0, value_cen)
                        series = Series([gauss(0, noise) for i in range(x.shape[2])])
                        #centroid
                        series_cen = Series([gauss(0, noise) for i in range(x.shape[2])])
                        
                        x[batch_idx] = x[batch_idx]+series
                        x_cen[batch_idx] = x_cen[batch_idx]+series_cen
                        
                        
                        value = x[batch_idx].mean()                        
                        series = Series([gauss(0, value) for i in range(x_global.shape[2])])
                        x_global[batch_idx] = x_global[batch_idx]+series
                        
                        #global cen
                        value_cen = x_cen[batch_idx].mean()
                        series_cen = Series([gauss(0, noise) for i in range(x.shape[2])])
                        x_cen_global[batch_idx] = x_cen_global[batch_idx]+series_cen

                    
                    outputs, atts, atts_g, atts_st = self.model(x, x_cen, x_global, x_cen_global,x_param, x_timel, x_timeg)
                    
                    loss = self.criterion(outputs, y)
                    loss.backward()
                    self.optimizer.step()

                    _, predicted = torch.max(outputs, 1)
                    correct += (predicted == y).sum().float().cpu().item()

                    self.experiment.log_metric("loss", loss.cpu().item(), step=steps_iter)
                    self.experiment.log_metric("accuracy", float(correct / total), step=steps_iter)
                    accuracy_step = self.experiment.log_metric("accuracy", float(correct / total), step=steps_iter)
                    print('===== train =====')
                    print('step ', steps_iter_val)
                    loss_step = self.experiment.log_metric("loss", loss.cpu().item(), step=steps_iter)
                    print('loss ', loss, 'loss.cpu.item', loss.cpu().item(), ' accuracy ', float(correct / total))
                    steps_iter = steps_iter + 1


            if validation:
                with self.experiment.validate():
                    with torch.no_grad():
                        val_correct = 0.0
                        val_total = 0.0

                        self.model.eval()
                        for x_val, x_cen_val,x_val_global, x_cen_val_global, x_val_param, x_val_timel, x_val_timeg, y_val in loader["val"]:
                            val_total += y_val.shape[0]
                            x_val = x_val.to(self.device) if isinstance(x_val, torch.Tensor) else [i_val.to(self.device) for i_val in x_val]
                            x_cen_val = x_cen_val.to(self.device) if isinstance(x_cen_val, torch.Tensor) else [i_val.to(self.device) for i_val in x_cen_val]
                            x_val_param = x_val_param.to(self.device) if isinstance(x_val_param, torch.Tensor) else [i_val.to(self.device) for i_val in x_val_param]
                            

                            x_val_global = x_val_global.to(self.device) if isinstance(x_val_global, torch.Tensor) else [i_val.to(self.device) for i_val in x_val_global]
                            x_cen_val_global = x_cen_val_global.to(self.device) if isinstance(x_cen_val_global, torch.Tensor) else [i_val.to(self.device) for i_val in x_cen_val_global]

                            y_val = y_val.to(self.device)

                            
                            val_output, atts_val, atts_g_val, atts_st_val = self.model(x_val, x_cen_val, x_val_global, x_cen_val_global, x_val_param)
                            
                            val_loss = self.criterion(val_output, y_val)
                            _, val_pred = torch.max(val_output, 1)
                            val_correct += (val_pred == y_val).sum().float().cpu().item()

                            self.experiment.log_metric("loss", val_loss.cpu().item(), step=steps_iter_val)
                            self.experiment.log_metric("accuracy", float(val_correct / val_total), step=steps_iter_val)
                            accuracy_step = self.experiment.log_metric("accuracy", float(correct / total), step=steps_iter_val)
                            print('==== validation ====')
                            print('step ', steps_iter_val)
                            loss_step = self.experiment.log_metric("loss", loss.cpu().item(), step=steps_iter_val)
                            print('loss ', loss_step, 'loss ', val_loss.cpu().item(), ' accuracy ', float(val_correct / val_total))
                            steps_iter_val = steps_iter_val + 1

                        print("\033[33m" + "Evaluation finished. " + "\033[0m" + "Accuracy: {:.4f}".format(float(val_correct / val_total)))


            pbar.close()

        return [], []
    
    
