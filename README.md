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
Once all the aforementionned requirements are satisfied, one can easily train a new GATGNN by running __train.py__ in the terminal along with the specification of the appropriate flags. At the bare minimum, using --property to specify the property and --data_src to identify the dataset (CGCNN or MEGNET) should be enough to train a robust GATGNN model.