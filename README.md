# Binding_affinity_prediction

## Introduction :

 Accurate prediction of drug target interactions(DTI) is one of the most crucial steps in the early phase of new drug discovery. However experimentally, this step is extremely expensive and time consuming. The following table gives a general scale of cost and time required for discovering a new drug:




 ![image](https://github.com/Gilgamesh60/Binding_affinity_predictor/assets/104096164/2af77296-32a4-4056-9409-0e17a3916139)


Binding affinity is one of the most important measures for drug-target interaction and can help us design drugs that selectively binds to their target. 
One of the most popular computational methods for binding affinity prediction is molecular docking. It greatly reduces the computational costs and gives predictions with good accuracy. Unfortunately, this accuracy is still not sufficient to be practically used for drug discovery.There is also a prerequisite of already knowing the location of active sites. 

So the aim of this project is to use deep learning techniques to improve the performance of DTI calculations and try to provide an alternative to conventional methods like molecular docking and experimentaion.



## Dataset :

One of the main challenges in using deep learning for this problem is limited but high complex datasets. For this project, I am using the PDBBINDv2016 refined and general minus refined datasets.The refind dataset contains 4057 protein(target)-ligand(drug) complexes in total. This dataset was made by compiling the protein-ligand complexes with better quality out of the general dataset.General minus refined dataset consists of 9228 complexes. Both of them are similar in their structure. 
Refined dataset has total 4057 folders with each folder representing a protein-ligand complex. Protein structure is stored in a PDB file format. Ligand structure is stored in a SDF or MOL2 file format. Index folder summarizes the basic information about the protein-ligand complexes including the binding affinity of complexes.



## Method :

Graph neural networks(GNNs) have proved to be one of the most prominent models in the field of drug discovery. In this approach, I am planning to use Graph Attention Networks (GAT) for binding affinity prediction.

**1. Interactions graph :-**

![image](https://github.com/Gilgamesh60/Binding_affinity_predictor/assets/104096164/5cac35f6-4a36-4d1c-86a8-d366a37daa51)


First we need to build an interactions graph between proteins and ligands. First we will construct an individual protein and ligand graph from the given pdb and sdf files. This graph contains following information :

        * node_feat : Features of graph nodes.Contains atomic number,degrees of freedom,valency and aromatic nature of all the atoms in the molecule. Used to create the initial feature matrix (X)
        
        * num_of_nodes : Number of nodes in the molecule graph.
        
        * edge_index : Contains the information about bond connections.Used for creating the adjacency matrix(A)
        
        * edge_pos : Contains the information about intermolecular bond distance. Can be used as edge_feature.
