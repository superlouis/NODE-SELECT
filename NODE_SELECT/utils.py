import numpy  as np
import pandas as pd
import torch, time, warnings, sys, argparse

import torch.nn.functional as F
from sklearn.model_selection import train_test_split, StratifiedKFold

from tabulate import tabulate
from torch_geometric.datasets import Planetoid, CitationFull, Coauthor, Amazon
import matplotlib.pyplot as plt
import itertools, pickle

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    
def set_up_model(framework):
    model_name      = ['NSGNN','GCN','GAT','GRAPHSAGE','MLP']
    model_reference = [[1,'X'],[0,1],[0,3],[0,2],[0,7]]
    reference       = dict(zip(model_name,model_reference))
    return reference[framework]

def load_metrics(name=''):
    saved_metrics = pickle.load(open(f'RESULTS/metrics_{name}.pickle', "rb", -1))
    return saved_metrics

def plot_metrics_training(metrics,title,name,index_choice=1):
    if index_choice == 1:
        training    = metrics.training_loss1
        validation  = metrics.valid_loss1
    else:
        training    = metrics.training_loss2
        validation  = metrics.valid_loss2
    epochs          = range(1,len(training)+1)
    idx_min         = np.argmin(validation)+1
    plt.plot(epochs, training,color='green',label = 'Training')
    plt.plot(epochs, validation,color='red',label = 'Validation')
    plt.grid()
    # plt.axvline(x = idx_min, color ='black',label = 'Early-stopping') 
    plt.legend(loc='best',fontsize=18)
    plt.xlabel('Epochs',fontsize=18)    
    plt.xticks(fontsize=18)
    plt.ylabel('Loss',fontsize=15)    
    plt.yticks(fontsize=15)         
    plt.tight_layout()    
    plt.title(title,fontsize=18)
    plt.gca().set_ylim(0,2.5)
    plt.tight_layout()
    plt.savefig(f'{name}')

def make_masks(data_y, random_seed, mode='training-first', split='stratified'):
    values       = data_y.cpu()
    num_nodes    = values.size(0)
    mask_values0 = torch.zeros(num_nodes).bool()
    mask_values1 = torch.zeros(num_nodes).bool()
    mask_values2 = torch.zeros(num_nodes).bool()

    if split    == 'stratified':
        skf          = StratifiedKFold(5, shuffle=True, random_state=random_seed)
        idx          = [torch.from_numpy(i) for _, i in skf.split(values, values)]

        if mode     == 'training-first' :
            train   = torch.cat(idx[:3], dim=0)
            valid   = idx[4]
            test    = idx[5]
        elif mode   == 'testing-first'  :
            train   = idx[0]
            valid   = idx[1]
            test    = torch.cat(idx[2:], dim=0)
    elif split  == 'random':
        all_masks   = np.arange(num_nodes)

        if mode     == 'training-first':
            train,test_V   = train_test_split(all_masks,train_size=0.6,random_state=random_seed)
            valid,test     = train_test_split(test_V,test_size=0.5,random_state=random_seed)
        elif mode   == 'testing-first':
            train_V,test   = train_test_split(all_masks,test_size=0.6,random_state=random_seed)
            train,valid    = train_test_split(train_V,test_size=0.5,random_state=random_seed)

    mask_values0[train] = True
    mask_values1[test]  = True
    mask_values2[valid] = True        

    return mask_values0, mask_values1, mask_values2 

def import_dataset(name='CORA'):
    root    = f'BENCHMARK/{name.upper()}/'
    if   name.upper() == 'CORA':
        dataset = Planetoid(root=root, name='CORA')
    elif   name.upper() == 'CORA-F':
        dataset = CitationFull(root=root, name='cora')        
    elif name.upper() == 'CITESEER':
        dataset = Planetoid(root=root, name='citeseer')
    elif name.upper() == 'PUBMED':
        dataset = Planetoid(root=root, name='PubMed')
    elif name.upper() == 'COAUTHOR-P':
        dataset = Coauthor(root=root, name='Physics')
    elif name.upper() == 'COAUTHOR-C':
        dataset = Coauthor(root=root, name='CS')        
    elif name.upper() == 'AMAZON-C':
        dataset = Amazon(root=root, name='Computers')    
    elif name.upper() == 'AMAZON-P':
        dataset = Amazon(root=root, name='Photo')    

    elif name.lower() == 'all':
        Planetoid(root=root, name='CORA')
        Planetoid(root=root, name='citeseer')
        CitationFull(root=root, name='cora')        
        Planetoid(root=root, name='PubMed')
        Coauthor(root=root, name='Physics')
        Coauthor(root=root, name='CS')
        Amazon(root=root, name='Computers')    
        Amazon(root=root, name='Photo')    
        exit()
    return dataset

def output_training(metrics_obj,epoch,estop_val,extra='---'):
    header_1, header_2 = 'NLL-Loss | e-stop','Accuracy | e-stop'

    train_1,train_2    = metrics_obj.training_loss1[epoch],metrics_obj.training_loss2[epoch]
    valid_1,valid_2    = metrics_obj.valid_loss1[epoch],metrics_obj.valid_loss2[epoch]
    
    tab_val = [['TRAINING',f'{train_1:.4f}',f'{train_2:.4f}%'],['VALIDATION',f'{valid_1:.4f}',f'{valid_2:.4f}%'],['E-STOPPING',f'{estop_val}',f'{extra}']]
    
    output = tabulate(tab_val,headers= [f'EPOCH # {epoch}',header_1,header_2],tablefmt='fancy_grid')
    print(output)
    return output

def live_plot(epoch, Training_list, Validation_list, watch=False,interval=0.2,extra_array=[]):
    if watch ==  True:
        if epoch >=1:
            plt.plot([epoch,epoch+1],[Training_list[epoch-1],Training_list[epoch]],'g-')
            plt.plot([epoch,epoch+1],[Validation_list[epoch-1],Validation_list[epoch]],'r-')
            if len(extra_array)>0:
                plt.plot([epoch,epoch+1],[extra_array[epoch-1],extra_array[epoch]],'b-')
            plt.pause(interval)
    else: pass

def torch_accuracy(tensor_pred,tensor_real):
    correct = torch.eq(tensor_pred,tensor_real).sum().float().item()
    return 100*correct/len(tensor_real)

def mess_up_dataset(dataset, num_noise):
    dataset       = dataset.to('cpu')
    actual_labels = torch.unique(dataset.y)
    actual_nodes  = np.arange(dataset.x.size(0)).reshape(-1,1)

    real_flags    = np.ones(dataset.x.size(0))
    fake_flags    = np.zeros(num_noise)
    flags         = np.hstack([real_flags,fake_flags])

    np.random.seed(num_noise)
    torch.manual_seed(num_noise)

    print('> Number of fake data: ',num_noise)

    fake_nodes    = np.arange(dataset.x.size(0),dataset.x.size(0)+num_noise)
    size_feat     = dataset.x.size(1)
    avg_connect   = int(dataset.edge_index.size(1)/dataset.x.size(0))
    # fake data
    fake_labels   = torch.tensor(np.random.choice(actual_labels,num_noise).reshape(-1))
    fake_feature  = torch.randn(num_noise,size_feat)

    # making fake edges
    real2fake     = np.random.choice(fake_nodes,size=(dataset.x.size(0),avg_connect)).reshape(-1)
    fake2real     = np.repeat(actual_nodes,avg_connect,axis=-1).reshape(-1)

    np_edge_index = dataset.edge_index.numpy()

    temp_TOP      = np.hstack((np_edge_index[0],fake2real))
    idx_sorting   = np.argsort(temp_TOP)
    TOP           = np.sort(temp_TOP)
    temp_bottom   = np.hstack([np_edge_index[1],real2fake])
    BOTTOM        = temp_bottom[idx_sorting]

    REAL_add      = np.vstack([TOP,BOTTOM])
    FAKE_add      = np.vstack([real2fake,fake2real])
    
    # all-together
    dataset.edge_index = torch.tensor(np.hstack([REAL_add,FAKE_add]))
    dataset.x          = torch.cat([dataset.x,fake_feature],dim=0)
    dataset.y          = torch.cat([dataset.y,fake_labels],dim=-1)
    dataset.flags      = torch.tensor(flags)

    return dataset
 

