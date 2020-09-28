import torch, pickle
import numpy as np 
from sklearn.metrics import accuracy_score as sk_acc
from sklearn.metrics import average_precision_score as sk_precision
import pandas as pd

class METRICS:
    def __init__(self,epoch,torch_criterion,torch_func,device,name_cols=[],name_='',classification=False):
        self.criterion         = torch_criterion
        self.metrics_name      = name_
        self.eval_func         = torch_func
        self.dv                = device
        self.training_measure1 = torch.tensor(0.0).to(device)
        self.training_measure2 = torch.tensor(0.0).to(device)
        self.valid_measure1    = torch.tensor(0.0).to(device)
        self.valid_measure2    = torch.tensor(0.0).to(device)

        self.training_counter  = 0
        self.valid_counter     = 0
        self.temp_counter      = 0

        self.training_loss1    = []
        self.training_loss2    = []
        self.valid_loss1       = []
        self.valid_loss2       = []
        self.duration          = []
        self.class_on_training = classification

    def __str__(self):
        x = self.to_frame() 
        return x.to_string()

    @property
    def dataframe(self):
        metrics_df = pd.DataFrame(list(zip(self.training_loss1,self.training_loss2,
                                           self.valid_loss1,self.valid_loss2,self.duration)),
                                           columns=['training_1','training_2','valid_1','valid_2','time'])
        return metrics_df

    def save_time(self,e_duration):
        self.duration.append(e_duration)

    def convert_prob2class(self,some_tensor):
        return torch.argmax(some_tensor,dim=-1)

    def __call__(self,which_phase,tensor_pred,tensor_true,measure=1):
        if measure == 1:
            if   which_phase == 'training'  : 
                loss         = self.criterion(tensor_pred,tensor_true)
                self.training_measure1 += loss
            elif which_phase == 'validation':
                loss         = self.criterion(tensor_pred,tensor_true)
                self.valid_measure1    += loss
        else:
            if self.class_on_training   : tensor_pred = self.convert_prob2class(tensor_pred)
            if   which_phase == 'training'  :
                loss         = self.eval_func(tensor_pred,tensor_true) 
                self.training_measure2 += loss
            elif which_phase == 'validation':
                loss         = self.eval_func(tensor_pred,tensor_true)
                self.valid_measure2    += loss
        return loss

    def reset_parameters(self,which_phase,epoch):
        if   which_phase == 'training'  :
            # AVERAGES
            t1 = self.training_measure1/(self.training_counter)
            t2 = self.training_measure2/(self.training_counter)

            self.training_loss1.append(t1.item())
            self.training_loss2.append(t2.item())
            self.training_measure1 = torch.tensor(0.0).to(self.dv)
            self.training_measure2 = torch.tensor(0.0).to(self.dv)
            self.training_counter  = 0 
        else:
            # AVERAGES
            v1 = self.valid_measure1/(self.valid_counter)
            v2 = self.valid_measure2/(self.valid_counter)

            self.valid_loss1.append(v1.item())
            self.valid_loss2.append(v2.item())
            self.valid_measure1    = torch.tensor(0.0).to(self.dv)
            self.valid_measure2    = torch.tensor(0.0).to(self.dv)      
            self.valid_counter     = 0 
    
    def save_info(self):
        with open(f'RESULTS/metrics_{self.metrics_name}.pickle','wb') as metrics_file:
            pickle.dump(self,metrics_file)

    def run_classification_measures(self,tensor1,tensor2,):
        tensor1,tensor2            = tensor1.cpu().numpy(),tensor2.cpu().numpy()
        accuracy                   = sk_acc(tensor1,tensor2)
        return accuracy*100

    def evaluation_results(self,tensor1,tensor2):
        results                = self.run_classification_measures(tensor1,tensor2)
        return results