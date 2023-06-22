from configure import *
from functions import *
import rdkit
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from rdkit.Chem.rdmolfiles import MolFromMol2File, MolFromPDBFile
from rdkit.Chem import SDMolSupplier
import torch
from torch_geometric.data import InMemoryDataset, Dataset, Data
class protein_ligand_dataset(InMemoryDataset):
    def __init__(self,data_dir, affinity_file, transform = None, pre_transform = None):
        self.binding_affinity = affinity_file
        self.data_dir = data_dir
        super(protein_ligand_Dataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def build(self):
        data_list = []

        complex_to_affinity = binding_affinity_extractor(self.binding_affinity)

        for i in tqdm(range(len(os.listdir(self.data_dir)))):
            complex_folder_name = os.listdir(self.data_dir)[i]

            if complex_folder_name=="readme" or complex_folder_name=='index':
              continue
            complex_id = complex_folder_name
            
            # ligand structure in sdf file
            ligand_file_path = os.path.join(self.data_dir, complex_folder_name, "{}_ligand.sdf".format(complex_id))
            # protein structure in pdb file
            protein_file_path = os.path.join(self.data_dir, complex_folder_name, "{}_protein.pdb".format(complex_id))        
            graph = interactions_graph(ligand_structure_file_path, protein_structure_file_path)
            binding_affinity = map_complex_to_affinity[pdb_id]
            data = Data()
            data.__num_nodes__ = graph["num_nodes"]
            data.x = torch.from_numpy(graph["node_feat"]).to(torch.float32)
            edge_index = torch.from_numpy(graph["edge_index"]).to(torch.long)
            data["edge_index_1"] = edge_index[:, : graph["num_covalent_bonds"]]
            data["edge_index_2"] = edge_index
            data["edge_attr"] = torch.from_numpy(graph["edge_attr"]).to(torch.float32)
            data.y = torch.Tensor([binding_affinity]).to(torch.float32)
            data_list.append(data)

        data, slices = self.collate(data_list)
        print("Done!")
        torch.save((data, slices), self.processed_paths[0])
