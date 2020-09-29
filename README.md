# NODE-SELECT Graph Neural Networks

This software package implements the work done in the paper "NODE-SELECT: A FLEXIBLE GRAPH NEURAL NETWORK BASED ON REALISTIC PROPAGATION SCHEME". This is the official Pytorch repository.

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
Once all the aforementionned requirements are satisfied, one can easily our codes using [the appropriate flags](./miscallaneous.txt). A set of the best hyper-parameters is provided [in this file](./NODE-SELECT_Configurations.pdf).

- Example-1. Train & evaluate a NODE-SELECT model of  on the bulk-modulus property using the CGCNN dataset.
```bash
python train.py --property bulk-modulus --data_src CGCNN
```

