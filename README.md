# NODE-SELECT Graph Neural Networks

This software package implements the work done in the paper "NODE-SELECT: A Graph Neural Network Based On A Selective Propagation Technique". This is the official Pytorch repository.  

![](front-pic.png)
<a name="installation"></a>
## Installation
Install any of the relevant packages if not already installed:
* Pytorch (tested on 1.4.0) - preferably version 1.4.0 or later
* Numpy   (tested on 1.19.2)
* Pandas  (tested on 1.1.2) 
* Scikit-learn (tested on 0.23.2) 
* Matplotlib (tested on 3.2.2)
* PyTorch-Geometric (tested on 1.4.3)
* Tabulate (tested on 0.8.7)
- Pytorch, Numpy, Pandas, Scikit-learn, Matplotlib, and Tabulate

```bash
pip install torch torchvision 
pip install numpy
pip install pandas
pip install scikit-learn
pip install matplotlib
pip install tabulate
```

- PyTorch Geometric [documentation](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html#installation)


<a name="usage"></a>
## Usage
Once all the aforementionned requirements are satisfied, one can easily our codes using [the appropriate flags](./miscellaneous.txt). A set of the best hyper-parameters is provided [in this file](./NODE-SELECT_Configurations.pdf), please refer to it.

- Example-1. Train & evaluate a NODE-SELECT (NSGNN) model on the Pubmed dataset: with a learning-rate of 0.01, weight-decay of 0.0005, and built with 5 layers (filters).

```bash
python main.py --benchmark pubmed --framework NSGNN --lr 0.01 --weight_decay 0.0005  --layers 5
```

- Example-2. Train & evaluate a GCN model on the Cora dataset: with a learning-rate of 0.01, weight-decay of 0.0005, and built with 2 layers.

```bash
python main.py --benchmark cora --framework GCN --lr 0.01 --weight_decay 0.0005  --layers 2
```

- Example-3. Train & evaluate a GAT model on the Amazon-Photos dataset: with a learning-rate of 0.005, weight-decay of 0.00005, 128 neurons (16 neurons\*8heads), built with 2 layers, and 8 attention-heads.

```bash
python main.py --benchmark amazon-p --framework GAT --lr 0.005 --weight_decay 0.00005  --layers 2 --heads 8 --neurons 16
```

## Citation
If you find our codes or project useful for your research, please cite our work:
```bash
@article{louis2022node,
  title={Node-select: a graph neural network based on a selective propagation technique},
  author={Louis, Steph-Yves and Nasiri, Alireza and Rolland, Fatima J and Mitro, Cameron and Hu, Jianjun},
  journal={Neurocomputing},
  year={2022},
  publisher={Elsevier}
}
```
