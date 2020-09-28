## Flags 
* --benchmark : benchmark dataset (default: cora)
    * choices are cora, citeseer, cora-f , pubmed, coauthor-p, coauthor-c, amazon-p, and amazon-c
* --framework : model choices (default: NSGNN)
    * choices are NSGNN, GCN, GAT, GRAPHSAGE, and MLP
* --layers: number of layers needed to construct your model (default:1)
* --neurons: number of neurons to use for hidden layers of your model. \*\*\*not needed for NSGNN
* --lr: learning rate (default:0.01)
* --num_splits: number of different data-splits to use for training and testing mode (default:10)
* --weight_decay: weight decay to use for Adam optmizer (default:0.0005)
* --heads: number of attention-heads to use for GAT \*\*\*only needed for GAT (default:8)
* --depth: propagation depth of NSGNN filter \*\*\*only needed for NODE-SELECT (default:1)
* --random: whether to randomize the seeds used for training/testing the model (default:False)


