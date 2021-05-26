from NODE_SELECT.baseline import *
from NODE_SELECT.model import *
from NODE_SELECT.utils import *
from NODE_SELECT.data  import *

# ARGS PARAMETERS
parser = argparse.ArgumentParser(description='NODE-SELECT Graph Neural Network')

# the dataset to use
parser.add_argument('--benchmark', default='cora',
                    choices=['cora','citeseer','cora-f','pubmed','coauthor-p','coauthor-c','amazon-p','amazon-c'],
                    help='benchmark dataset (default: cora')

# the GNN framework
parser.add_argument('--framework', default='NSGNN',choices=['NSGNN','GCN','GAT','GRAPHSAGE','MLP'],
                    help='model choices (default: NSGNN)')

# Learning Hyper-parameters
parser.add_argument('--lr',default=1e-2, type=float, help='learning rate')
parser.add_argument('--weight_decay',default=5e-4, type=float, help='weight decay to use for Adam optmizer')

# Model Parameters 
parser.add_argument('--layers', default =1, type=int,
                    help='number of layers needed to construct your model (default:1)')
parser.add_argument('--neurons',default=64, type=int,
                    help='number of neurons to use for hidden layers of your model. ***not needed for NSGNN')
parser.add_argument('--heads',  default=8, type=int, help='number of attention-heads to use  for GAT   ***only needed for GAT')
parser.add_argument('--depth',  default=1, type=int, help='propagation depth of NSGNN filter for NSGNN ***only needed for NODE-SELECT')

# Performance Parameters
parser.add_argument('--num_splits',default=10, type=int,
                    help='number of different data-splits to use for training and testing model')
parser.add_argument('--random', default=False, type=bool,  help ='whether to randomize the seeds used for training/testing the model')
parser.add_argument('--noise',  default=0.0,   type=float, help ='percentage of noise to add to dataset')


args = parser.parse_args(sys.argv[1:])

# GPU-SETTING & TRAINING SETTINGS
device        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loop_run      = args.num_splits

# DATA PARAMETERS
name_dataset  = args.benchmark
dataset       = import_dataset(name_dataset)
data          = dataset[0].to(device)

noise_pct     = args.noise
noise2add     = int(data.x.size(0)*noise_pct)

if noise2add >0:
    data      = mess_up_dataset(data, noise2add).to(device)

feat_in       = data.x.size(1)
feat_mid      = args.neurons
feat_out      = len(torch.unique(data.y))
NSGNN_depth   = args.depth
gat_heads     = args.heads

# HYPER-PARAMETERS
M_choice,M_0_model =  set_up_model(args.framework)
n_epoch            = 200
lr_rate            = args.lr
weight_decay       = args.weight_decay
n_layers           = args.layers

all_ACC            = []
all_MEMORY         = []
all_TIME           = []
all_SEEDS          = []

if args.random:
    the_seeds      = np.random.randint(50000,size=(loop_run,)) 
else:
    the_seeds      = np.unique([1053,116,89535,80,3,222,41,971,357,0,22222,468579,457,867,3794,6517,7245,4703])

for i in range(loop_run): 
    start_training = time.time()
    random_seed    = the_seeds[i]
    print('> DATA-Split # {} *** seed {}'.format(i+1,random_seed))    
    data.train_mask, data.test_mask, data.val_mask = make_masks(data.y,random_seed,'testing-first','stratified')

    # MODEL-CHOICE / TRAINING-PARAMETERS
    if   M_choice == 0:  
        model      = BaselineNet(feat_in,feat_mid,feat_out,  n_layer= n_layers, heads=gat_heads, architecture=M_0_model).to(device)
        optimizer  = torch.optim.Adam(model.parameters(), lr=lr_rate, weight_decay=weight_decay)
    elif M_choice == 1:  
        model      = NSGNN(feat_in, feat_out,  learners = n_layers, spread_L=NSGNN_depth , unifying_weight=False).to(device)  
        optimizer  = torch.optim.Adam(model.parameters(), lr=lr_rate, weight_decay=weight_decay)

    # METRICS-OBJECT INITIALIZATION
    M_name        = f'{args.framework}'
    metrics       = METRICS(n_epoch,F.nll_loss,torch_accuracy,device)
    for epoch in range(n_epoch):
        # TRAINING-STAGE
        model.train()
        optimizer.zero_grad()
        start_time = time.time()
        pred       = model(data)
        train_real = data.y[data.train_mask]
        train_pred = torch.argmax(pred[data.train_mask],dim=-1)
        loss       = metrics('training',pred[data.train_mask],train_real,1)
        _          = metrics('training',train_pred,train_real,2)
        loss.backward()
        optimizer.step()      
        metrics.training_counter+=1
        metrics.reset_parameters('training',epoch)                    

        # VALIDATION-STAGE
        model.eval()
        pred       = model(data)
        valid_real = data.y[data.val_mask]
        valid_pred = torch.argmax(pred[data.val_mask]  ,dim=-1)
        _          = metrics('validation',pred[data.val_mask],valid_real,1)
        _          = metrics('validation',valid_pred,valid_real,2)
        metrics.valid_counter+=1
        metrics.reset_parameters('validation',epoch)

        end_time   = time.time()
        e_time     = end_time-start_time
        metrics.save_time(e_time)

        if metrics.valid_loss1[-1] <= min(metrics.valid_loss1):
            estop_val               = '@best: saving model...'
            metrics.temp_counter    = 0
        else:                      
            metrics.temp_counter   += 1
            estop_val = f'> {metrics.temp_counter} / {n_epoch}__________'

        if loop_run > 1: pass
        else:
            extra = f' '
            output_training(metrics,epoch,estop_val,extra=extra)
            live_plot(epoch, metrics.training_loss1, metrics.valid_loss1, watch=True,interval=0.05)

        if epoch == n_epoch-1:
            mem_size = torch.cuda.max_memory_reserved(device)*1e-9
            all_MEMORY.append(mem_size)            

    # TESTING PHASE
    model.eval()
    outpred    = model(data)
    _, pred    = outpred.max(dim=1)
    test_real  = data.y[data.test_mask] 
    test_pred  = pred[data.test_mask]  
    acc        = metrics.evaluation_results(test_pred,test_real)

    # saving training-plot
    if loop_run > 1: pass
    else:
        title   = f'MODEL-{args.framework}---acc:{acc*100:.3f}---green (Val),red (Train), blue (mean-size)'
        name    = f'RESULTS/MODEL-{args.framework}-plot.png'

    # saving prediction 
    print('> Accuracy: {:.3f}%---------Memory: {:.3f} gb'.format(acc,mem_size))
    if M_choice != 0:
        # INDIVIDUAL PREDICTION
        for i in range(n_layers):
            out = torch.argmax(model.outputs[i][data.test_mask],dim=-1)
            temp_cor   = (out.eq(data.y[data.test_mask]).sum().item())
            temp_acc   = temp_cor / data.test_mask.sum().item()
            print(f'    Filter {i+1:<3} acc: {100*temp_acc:.3f}     V*: {model.leader_info[i]:.2f}')

    all_ACC.append(acc)
    all_SEEDS.append(random_seed)

    stop_time = time.time()
    the_time  = stop_time-start_training
    all_TIME.append(the_time)
    print('-'*40)

report_df = pd.DataFrame(list(zip(all_SEEDS,all_ACC,all_MEMORY,all_TIME)),columns=['Seeds','Accuracy','Memory','Time'])
print('='*40);print(report_df,'\n')
print(f'Mean Accucacy: {report_df.Accuracy.mean():.2f} +/- {report_df.Accuracy.std():.2f} ---- Avg. Memory: {report_df.Memory.mean():.3f} gb.')
