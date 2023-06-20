# Binding_affinity_prediction



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

!python run.py
        ----ligand_file "Put the ligand.sdf file path here"
        ----protein_file "Put the protein.pdb file path here"
```

## Introduction :

This project is an effort to incorporate machine learning in my chemical engineering domain.

Accurate prediction of drug target interactions(DTI) is one of the most crucial steps in the early phase of new drug discovery. However experimentally this step is extremely expensive and time consuming. The following table gives a general scale of cost and time required for discovering a new drug:




 ![image](https://github.com/Gilgamesh60/Binding_affinity_predictor/assets/104096164/2af77296-32a4-4056-9409-0e17a3916139)


Binding affinity is one of the most important measures for drug-target interaction and can help us design drugs that selectively binds to a specific target. 
One of the most popular computational methods for binding affinity prediction is molecular docking. It greatly reduces the costs and gives predictions with good accuracy. Unfortunately, this accuracy is still not sufficient to be practically used for drug discovery.There is also a prerequisite of already knowing the location of active sites. 

So the aim of this project is to use deep learning techniques to improve the performance of DTI methods and try to provide an alternative to conventional methods like molecular docking and experimentaion.

The model implemented here is completely based on the method mentioned in the paper [Predicting drug-target interaction using 3D structure-embedded graph representations from graph neural networks](https://pubs.acs.org/doi/pdf/10.1021/acs.jcim.9b00387)

## Dataset :

One of the main challenges in using deep learning for this problem is limited but high complex datasets. For this project, I am using the PDBBINDv2016 refined and general minus refined datasets.The refind dataset contains 4057 protein(target)-ligand(drug) complexes in total. This dataset was made by compiling the protein-ligand complexes with better quality out of the general dataset.General minus refined dataset consists of 9228 complexes. Both of them are similar in their structure. 
Refined dataset has total 4057 folders with each folder representing a protein-ligand complex. Protein structure is stored in a PDB file format. Ligand structure is stored in a SDF and MOL2 file format. Index folder summarizes the basic information about the protein-ligand complexes including the binding affinity of complexes.

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

I am using a graph attention mechanism. This mechanism combines the attention mechanism used in NLP in the graph neural networks. Idea is to amplify the more important features and downgrade the less important features. For eg. In a sentence "Children are playing on the ground", word "ground" should pay more "attention" to the words like "Children" and "playing" than words like "the","on". Similarly in our case we want to give more "attention" to the important protein-ligand intermolecular interactions. The attention mechanism is based on the legendary research paper ["Attention is all you need"](https://arxiv.org/abs/1706.03762). Please check it out if you are interested.

Input: The input of our graph attention model is the set of node features: $\mathbf{X_{\text{in}}} = \{\mathbf{x_1}, \dots, \mathbf{x_N}\}$ with $\mathbf{x_i} \in \mathbb{R}^F$ ($F$ is the number of features, $N$ is the number of nodes) and adjacency matrix **A** which keeps tracks of the edge coordinates. 

Each node feature is transformed by using a learable weight matrix $W \in \mathbb{R}^{F \times F}$:  $$\mathbf{x_i} = W\mathbf{x_i}$$

Compute attention coefficient (the importand of $i^{th}$ node feature to $j^{th}$ node feature):  $$e_{ij} = e_{ji} = \mathbf{x}^{T}_i \mathbf{E} \mathbf{x}_j + \mathbf{x}^{T}_j \mathbf{E} \mathbf{x}_i$$

with $\mathbf{E} \in \mathbb{R}^{F \times F}$ is a learnable matrix, only compute $e_{ij}$ if $\mathbf{A_{ij}} = \mathbf{A_{ji}} >0 $

As mentioned in the original attention paper I will use a softmax function to normalize the attention coefficients.By doing this,all the coefficients of a node will be transformed in a way such that all coefficients sum up to 1: 
$$a_{ij} = \frac{\exp(e_{ij})}{\sum_{j \in N(i)} \exp(e_{ij})} \mathbf{A_{ij}}$$

Finally update: 
$$\hat{\mathbf{x_i}} = \sum_{j \in N(i)} a_{ij} \mathbf{x_j}$$ 

Here I am trying to do a graph level prediction, so we need to compile aggregated features from all nodes. Here I am using a simple minimum-mean pooling to concatenate all features.

As mentioned in the paper, I am first calculating the prediction using the primary adjacency matrix . The output of this is x1. Then calculate using secondary adjacency matrix. The output of this is x2.The final output for a node feature is just simply **x2 - x1**. By doing this we let our model learn the differences between the individual structures and the combined complex structure.

Source code : [src](https://github.com/Gilgamesh60/Binding_affinity_predictor/blob/main/src)

##  Results : 

These are all the results obtained from visualization of dataset and model performance testing:

### i) Random protein sample structure :
![image](https://github.com/Gilgamesh60/Binding_affinity_predictor/assets/104096164/c8967515-8d4e-48b5-9d7d-2882b25ccef1)

### ii) Protein binding pocket structure :
![image](https://github.com/Gilgamesh60/Binding_affinity_predictor/assets/104096164/5c554b38-cda0-43a2-b123-85d127638952)

### iii) Random Ligand sample structure :

![image](https://github.com/Gilgamesh60/Binding_affinity_predictor/assets/104096164/816240d8-d7f8-4313-bf69-40be1a3b7369)


### iv) Different types of atoms in a random protein sample :

![image](https://github.com/Gilgamesh60/Binding_affinity_predictor/assets/104096164/24cd3984-2937-458c-ac86-e6c5326f15a3)

### v) Atom type wise charge distribution in the protein sample

![image](https://github.com/Gilgamesh60/Binding_affinity_predictor/assets/104096164/623a3ba4-0d96-46d5-92f1-9d8f375aba32)







