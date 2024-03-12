# Mesh classification with Graph Neural Network

This repository hold my final project for the _Neural Networks and Deep Learning (2023)_ course at UniPD.

The project presents an simple, original Message Passing Graph Neural Network (GNN) architecture that processes input graphs of arbitrary size representing a 3D mesh of an object.
With the help of Torch Geometric's framework for developing GNN models, the goal was to see if a simple architecture that updates nodes embeddings without altering the graph topology was able to extract nodes features meaningful enough to perform object classification succesfully.

The pipeline is simple:

1. Load the dataset (ModelNet10/40 was used for benchmarking); each mesh is a graph, with nodes being the vertices of the mesh and the edges given by the triangular connectivity
2. Preprocess the dataset via a custom transform defined in `transform.py`
3. Train the model defined in `model.py`

### Usage

Install working `conda` environment:

```bash
conda env create --name envname --file=environment.yml; conda activate envname
```

define training parameters in `config.json`, then launch training with

```bash
python train.py
```

TensorBoard support is included for metrics monitoring.
