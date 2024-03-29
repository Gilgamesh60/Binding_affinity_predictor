import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit.Chem.rdmolfiles import MolFromMol2File, MolFromPDBFile
from rdkit.Chem import SDMolSupplier


#function to extract binding affinities of all the complexes in the PDBbindv2016 dataset
def binding_affinity_extractor(filename):
    map_complex_to_affinity = {}
    with open(filename) as f:
        for line in f:
            if line.startswith("#"):
                continue
            line = line.split()
            pdb_id, binding_affinity = line[0], line[3]
            map_complex_to_affinity[pdb_id] = float(binding_affinity)
    return map_complex_to_affinity

  
  
#Building a graph from a particular structure-protein/biomolecule(target) or its ligand partner(drug)
def structure_to_graph(structure):
  # atom
  #GetAtoms() can be used to get the info of all the atoms in protein
  #Each atom class contains atomic number,degrees of freedom,valency and aromatic nature.
  atoms_feature_list =  [[atom.GetAtomicNum(),atom.GetDegree(),atom.GetTotalNumHs(includeNeighbors = True), atom.GetImplicitValence(),int(atom.GetIsAromatic())] for atom in structure.GetAtoms()]
  node_features = np.array(atoms_feature_list, dtype = np.float64)
  #GetConformer() - Get that paticular conformer of the protein
  c = structure.GetConformer()
  coordinates = [[c.GetAtomPosition(atom_index)[i] for i in range(3)] for atom_index in range(structure.GetNumAtoms())]
  node_positions = np.array(coordinates, dtype = np.float64)

  # bond
  #GetNumBonds() - number of bonds in the protein
  if structure.GetNumBonds() > 0:
        edge_list = []
        #GetBonds() - all bonds in the protein.This bond contains the beginning and last atom of the bond
        for bond in structure.GetBonds():
            atom_u, atom_v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edge_list.append((atom_u, atom_v))
            edge_list.append((atom_v, atom_u))

        edge_index = np.array(edge_list, dtype = np.int64).T
  else:
    edge_index = np.empty((2, 0), dtype = np.int64)
  graph = {"node_feat": node_features,
            "num_nodes": node_features.shape[0],
            "edge_index": edge_index,
            "node_positions": node_positions}
  return graph

#This function is used to calculate covalent interactions
def interactions_graph(ligand_file, protein_file):
    ligand = next(SDMolSupplier(ligand_file_path, sanitize = False))
    if ligand is None:
        return None
    protein = MolFromPDBFile(protein_structure_file_path, sanitize = False)
    if protein is None:
        return None
    ligand_graph = structure_to_graph(ligand)
    protein_graph = structure_to_graph(protein)
    node_features = np.concatenate([ligand_graph["node_feat"], protein_graph["node_feat"]], axis = 0)
    num_nodes = ligand_graph["num_nodes"] + protein_graph["num_nodes"]
    intermolecular_bonds, weight = [], []
    for latm_id in range(ligand_graph["num_nodes"]):
        for patm_id in range(protein_graph["num_nodes"]):
            ligand_atom_pos = ligand_graph["node_positions"][latm_id]
            protein_atom_pos = protein_graph["node_positions"][patm_id]
            distance = sum((ligand_atom_pos - protein_atom_pos) ** 2)
            if distance < 5:
                intermolecular_bonds.append((ligand_atom_id, protein_atom_id + ligand_graph["num_nodes"]))
                intermolecular_bonds.append((protein_atom_id + ligand_graph["num_nodes"], ligand_atom_id))
                weight.append(distance)
                weight.append(distance)

    if len(intermolecular_bonds) > 0:
        intermolecular_edge_index = np.array(intermolecular_bonds, dtype = np.int64).T
        intermolecular_edge_weight = np.array(weight, dtype = np.float64).reshape(1, len(intermolecular_bonds))
    else:
        intermolecular_edge_index = np.empty((2, 0), dtype = np.int64)
        intermolecular_edge_weight = np.empty((1, 0), dtype = np.float64)

    num_covalent_bonds = ligand_graph["edge_index"].shape[1] + protein_graph["edge_index"].shape[1]
    edge_index = np.concatenate([ligand_graph["edge_index"], protein_graph["edge_index"] + ligand_graph["num_nodes"],
                                intermolecular_edge_index], axis = 1)
    edge_weight = np.concatenate([np.array([[1] * ligand_graph["edge_index"].shape[1]]), np.array([[1] * protein_graph["edge_index"].shape[1]]),
                                intermolecular_edge_weight], axis = 1).reshape(edge_index.shape[1])
    node_positions = np.concatenate([ligand_graph["node_positions"], protein_graph["node_positions"]], axis = 0)

    graph = {
        "node_feat": node_features,
        "num_nodes": num_nodes,
        "num_covalent_bonds": num_covalent_bonds,
        "edge_index": edge_index,
        "edge_attr": edge_weight,
        "node_positions": node_positions,
    }

    return graph
