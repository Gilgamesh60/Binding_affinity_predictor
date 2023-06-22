# Binding_affinity_prediction



## Introduction :

This project is an effort to incorporate machine learning in my chemical engineering domain.

Accurate prediction of drug target interactions(DTI) is one of the most crucial steps in the early phase of new drug discovery. However experimentally this step is extremely expensive and time consuming. The following table gives a general scale of cost and time required for discovering a new drug:




 ![image](https://github.com/Gilgamesh60/Binding_affinity_predictor/assets/104096164/2af77296-32a4-4056-9409-0e17a3916139)


Binding affinity is one of the most important measures for drug-target interaction and can help us design drugs that selectively binds to a specific target. 
One of the most popular computational methods for binding affinity prediction is molecular docking. It greatly reduces the costs and gives predictions with good accuracy. Unfortunately, this accuracy is still not sufficient to be practically used for drug discovery.There is also a prerequisite of already knowing the location of active sites. 

So the aim of this project is to use deep learning techniques to improve the performance of DTI methods and try to provide an alternative to conventional methods like molecular docking and experimentaion.

The model implemented here is completely based on the method mentioned in the paper [Predicting drug-target interaction using 3D structure-embedded graph representations from graph neural networks](https://pubs.acs.org/doi/pdf/10.1021/acs.jcim.9b00387)



## Usage 

The repo is developed in such a way that you can use it for predicting DTI for your own complex. So you can use this repo commercially. You just need to follow these 3 simple steps.

i) Step 1 :- Clone the repo in google colab

```python
!git clone "https://github.com/Gilgamesh60/Binding_affinity_predictor/"
%cd "/content/Binding_affinity_predictor/"
```

ii) Step 2 :- Download all the required packages.

```python
!pip install -r requirements.txt
```

iii)Step 3 :- Before this you need to have your protein and ligand structures stored in .pdb and .sdf file format respectively. After that run this code

```python

!python run.py  -l "Put the ligand.sdf file path here"
        -p "Put the protein.pdb file path here"
```

## Dataset :

One of the main challenges in using deep learning for this problem is - limited but highly complex datasets. For this project, I am using the PDBBINDv2016 "refined" and "general minus refined" datasets.The refind dataset contains 4057 protein(target)-ligand(drug) complexes in total. This dataset was made by compiling the protein-ligand complexes with better quality out of the general dataset.General minus refined dataset consists of 9228 complexes. Both of these datasets are similar in their structure. 
Refined dataset has total 4057 folders with each folder representing a protein-ligand complex. Protein structure is stored in a PDB file format. Ligand structure is stored in a SDF and MOL2 file format.There is also a pdb file which stores the structure of protein pocket. Index folder summarizes the basic information about the protein-ligand complexes including the binding affinity of complexes.

How the dataset calculates binding affinity is also very interesting. Most commonly the binding affinity for such complexes is calculated as $-log(K_i/K_a)$ where $K_i$ is the inhibition constant which represents the concentration of ligand required to occupy 50% of the receptor sites when no competing ligand is present. Smaller the value of $K_i$,
greater is the binding affinity. $K_a$ is the equilibrium association constant and it represents how tightly a ligand binds to a protein. Basically it's the equilibrium constant for the reaction : L + P ⇄  LP . Higher the value of $K_a$, greater is the binding affinity. 

Link for the dataset download : [PDBBINDv2016 refined database](https://drive.google.com/drive/folders/1s3i9rIPzQAD2OqEkE4qwVVPsuc7UT0Ol?usp=sharing)

Link for visualization of dataset : [Dataset visualization](https://github.com/Gilgamesh60/Binding_affinity_predictor/blob/main/visualize/dataset_visualization.ipynb)


## Approach :

Graph neural networks(GNNs) have proved to be one of the most prominent models in the field of drug discovery. In this approach, I am planning to use Graph Attention Networks (GAT) for binding affinity prediction.

**1. Interactions graph :-**

![image](https://github.com/Gilgamesh60/Binding_affinity_predictor/assets/104096164/5cac35f6-4a36-4d1c-86a8-d366a37daa51)


Building a protein-ligand interaction graph given their pdb and sdf files is probably the first and most important step for our project. The 3D atomic coordinates of molecules contain full structural information. For effective representation,molecular graphs in which atoms and chemical bonds correspond to nodes and edges respectively can be utilized for proteins and ligands. To accurately predict the binding affinity of protein-ligand complex,it is important to accurately take into account various types of intermolecular interactions. The paper that I am following, proposes a novel approach that directly incorporates the 3D structural information on a protein-ligand binding pose.
There are 3 main forces acting between protein and ligand - covalent,intermolecular vanderwaal and electrostatic. Here for the sake of simplicity, I am only considering the covalent and intermolecular forces. Vanderwaal forces between atoms **i** and **j** depend inversely on the distance between those two atoms. As described in a paper, we will use a simple normal function such that it decreases with increase in distance. The function parameters are set in such a way that it becomes **0** if **distance > 5 Å**.





**2. Graph construction :-**

In our case, graph can be defined as **G = (V, E ,A1 , A2)**. V is the set of nodes. E is a set of edges. A1 is the primary adjacency matrix and A2 is secondary adjaceny matrix. I am modelling the adjacency matrix exactly like described in the paper.

![image](https://github.com/Gilgamesh60/Binding_affinity_predictor/assets/104096164/61403248-3ef1-474c-86be-9e47650896bb)


![image](https://github.com/Gilgamesh60/Binding_affinity_predictor/assets/104096164/5e6f49c3-c2d6-4dcb-a67f-74c220c43f6a)


Primary adjacency matrix(A1) just represents the intramolecular covalent bonds. Secondary adjacency matrix(A2) contains not only intramolecular covalent bonds but also the intermolecular vanderwaal bonds. 

Structure of the final protein-ligand interaction graph is  :

        * node_feat : Features of graph nodes.Contains atomic number,degrees of freedom,valency and aromatic nature of all the atoms in the molecule. Used to create the initial feature matrix (X).
        
        * num_of_nodes : Number of nodes in the graph. 
        
        * edge_index1 : Used for creating adjacency matrix - A1 and A2.Contains the information about the covalent connections.
        
        * edge_index2 : Used for creating adjacency matrix - A1 and A2 .Contains the information about the intermolecular vanderwaal connections.
        
        * edge_weight : Contains the Vanderwaal bond strength calculated using intermolecular distance. Used for creating the secondary adjacency matrix - A2.



**3. Model Architecture :-** 

I am using a graph attention mechanism. This mechanism combines the attention mechanism used in NLP in the graph neural networks. Idea is to amplify the more important features and downgrade the less important features. For eg. In a sentence "Children are playing on the ground", word "ground" should pay more "attention" to the words like "Children" and "playing" than words like "the" and "on". Similarly in our case we want to give more "attention" to the important protein-ligand intermolecular interactions. The attention mechanism is based on the legendary research paper [Attention is all you need](https://arxiv.org/abs/1706.03762). Please check it out if you are interested.

Before starting the explaination on how attention is integrated with GNNs, first I will try to explain the working of gnns in short.I will also try to justify why I have used GNNs for this project. 

Fundamental idea of GNNs is to learn a suitable representation of graph data for neural networks. Given all the information about graphs like node features,node connections stored in adjacency matrix, a GNNs outputs something new called as node embedding for each of the nodes. These node embeddings contain the structural and feature information about the other nodes in the graph. So each node is aware of the context of other nodes in the graph.This is achieved by message passing layers which are the main building blocks of GNNs

In general,the formula for messaging passing can be represented as - 
$$h_u^k+1 = UPDATE^k(h_u^k,AGGREGATE^k({h_v^k,\forall v \in N(u)}))$$
The AGGREGATE function gathers all the structural and feature data from the other nodes in the graph. Then based on node embeddings and this aggregated information update function simply produces the new node embeddings. As evident by the formula,it is an iterative formula. Based on how you define your AGGREGATE and UPDATE functions, there are many varients of GNNs.
Now this is where the attention mechanism comes into play.In graph attention networks,importance of the features of the neighbour state is considered for the aggregation. As a result the importance of the features of the neighbour state is used as a factor in AGGREGATE function.
Before going into how exactly I plan to use this model for my project, I want to briefly go throught the architecture of graph attention networks. If you are interested please go through this [paper](https://arxiv.org/pdf/1710.10903.pdf).

Input: The input of our graph attention model is the set of node features: $\mathbf{X_{\text{in}}} = \{\mathbf{x_1}, \dots, \mathbf{x_N}\}$ with $\mathbf{x_i} \in \mathbb{R}^F$ ($F$ is the number of features, $N$ is the number of nodes) and adjacency matrix **A** which keeps tracks of the edge coordinates. 

The new node embeddings are obtained by multiplying the current node embeddings(initially $X*A*$ where A is node feature matrix and A is the adacency matrix) by a learnable weight matrix $W \in \mathbb{R}^{F \times F}$:  $$\mathbf{x_i} = W\mathbf{x_i}$$

The basic idea is to learn how important node j's features are for node i which is called as attention coefficient($e_{ij}$). 
In the paper, they have calculated attention coefficient as :  $$e_{ij} = e_{ji} = \mathbf{x}^{T}_i \mathbf{W} \mathbf{x}_j + \mathbf{x}^{T}_j \mathbf{W} \mathbf{x}_i$$

Here $\mathbf{W} \in \mathbb{R}^{F \times F}$ is a learnable matrix. We don't want our attention coefficient to be negate. So we will only compute $e_{ij}$ if $\mathbf{A_{ij}} = \mathbf{A_{ji}} >0 $

Finally as mentioned in the original attention paper I will use a softmax function to normalize the attention coefficients.By doing this,all the coefficients of a node will be transformed in such way that all coefficients will always sum up to 1: 
$$a_{ij} = \frac{\exp(e_{ij})}{\sum_{j \in N(i)} \exp(e_{ij})} \mathbf{A_{ij}}$$

Aggregation is now complete.Now we just need to update: 
$$\hat{\mathbf{x_i}} = \sum_{j \in N(i)} a_{ij} \mathbf{x_j}$$ 

In this project I am trying to do a graph level prediction, so we need to compile features from all nodes so that we can feed that output to a neural network. Here I am using a simple global add pooling to concatenate all features.Other intelligent pooling methods might give better results. 
As mentioned in the paper, I am first calculating the prediction using the primary adjacency matrix . The output of this is x1. Then calculate using secondary adjacency matrix. The output of this is x2.The final output for a node feature is just simply **x2 - x1**. The aim is to let our model learn the differences between the individual structures and the combined complex structure.

Link for source code : [Source](https://github.com/Gilgamesh60/Binding_affinity_predictor/blob/main/src)



## Dataset visualization results : 

### i) Random protein sample structure :
![image](https://github.com/Gilgamesh60/Binding_affinity_predictor/assets/104096164/c8967515-8d4e-48b5-9d7d-2882b25ccef1)

### ii) Protein binding pocket structure :
![image](https://github.com/Gilgamesh60/Binding_affinity_predictor/assets/104096164/5c554b38-cda0-43a2-b123-85d127638952)

### iii) Random Ligand sample structure :

![image](https://github.com/Gilgamesh60/Binding_affinity_predictor/assets/104096164/816240d8-d7f8-4313-bf69-40be1a3b7369)


### iv) Different types of atoms in a random ligand sample :

![image](https://github.com/Gilgamesh60/Binding_affinity_predictor/assets/104096164/24cd3984-2937-458c-ac86-e6c5326f15a3)

### v) Atom type wise charge distribution in the ligand sample

![image](https://github.com/Gilgamesh60/Binding_affinity_predictor/assets/104096164/623a3ba4-0d96-46d5-92f1-9d8f375aba32)


## Model results 

### Results on training data

![training_plot](https://github.com/Gilgamesh60/Binding_affinity_predictor/assets/104096164/8cebe55e-a566-4a4f-93b3-0fb3930b850b)

### Results on test data 

![test_plot](https://github.com/Gilgamesh60/Binding_affinity_predictor/assets/104096164/c1632498-b39f-413d-b585-552a05fc3201)

### Confusion matrix for binary classification of complexes

![confusion_matrix](https://github.com/Gilgamesh60/Binding_affinity_predictor/assets/104096164/eac73f87-3d41-4425-b8fc-bd2e9f5485f3)

## Future works :

There is lot I want to do for this project :
1. I want to launch a webapp that can be used to predict binding affinity for any protein-ligand pair
2. I want to expand this repo to other protein-ligand datasets like KIBA,DAVIS and DUD-E.
3. One of the main problems here is the fact that the binding affinity is calculated under fixed circumstances. In the PDBBIND dataset, binding affinity is calculated under normal conditions (Solvent=Water, T=293 K, P=101.3 kPa, pH=7.4). But affinity is heavily dependent on the solvent, pH, temperature, dissolved salts, etc. So in order to use this model commercially, it is important to integrate such parameters in the model too. So I plan to look into this problem too.


