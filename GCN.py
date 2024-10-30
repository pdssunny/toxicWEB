from rdkit import Chem
from rdkit.Chem import AllChem
import torch
from torch_geometric.data import Data
import os
import re
import pickle
import json
from rdkit.Chem import PandasTools
import pandas as pd
from torch_geometric.data import InMemoryDataset
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GraphConv
from torch_geometric.nn import GATConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.loader import DataLoader
import sys
sys.path.append('.')

# atom
atom_vocab = [
    'H', 'Li', 'Na', 'K', 'Rb', 'Cs', 'Be', 'Mg', 'Ca', 'Sr', 'Ba', 'Sc',
    'Y', 'Ti', 'Zr', 'Hf', 'V', 'Nb', 'Ta', 'Cr', 'Mo', 'W', 'Mn', 'Tc',
    'Re', 'Fe', 'Ru', 'Os', 'Co', 'Rh', 'Ir', 'Ni', 'Pd', 'Pt', 'Cu', 'Ag',
    'Au', 'Zn', 'Cd', 'Hg', 'B', 'Al', 'Ga', 'In', 'Tl', 'C', 'Si', 'Ge', 'Sn',
    'Pb', 'N', 'P', 'As', 'Sb', 'Bi', 'O', 'S', 'Se', 'Te', 'Po', 'F', 'Cl', 'Br',
    'I', 'At', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er',
    'Tm', 'Yb', 'Lu', 'He', 'Ne', 'Ar', 'Kr', 'Xe', 'Rn', 'Fr', 'Ra', 'Ac', 
    'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es',  '*']

atom_vocab = {a: i for i, a in enumerate(atom_vocab)}

atomic_radius = {'H': 0.79, 'Li': 2.05, 'Na': 2.23, 'K': 2.77, 'Rb': 2.98, 'Cs': 3.34, 'Be': 1.4, 'Mg': 1.72,
                     'Ca': 2.23, 'Sr': 2.45, 'Ba': 2.78, 'Sc': 2.09,
                     'Y': 2.27, 'Ti': 2, 'Zr': 2.16, 'Hf': 2.16, 'V': 1.92, 'Nb': 2.08, 'Ta': 2.09, 'Cr': 1.85,
                     'Mo': 2.01, 'W': 2.02, 'Mn': 1.79, 'Tc': 1.95,
                     'Re': 1.97, 'Fe': 1.72, 'Ru': 1.89, 'Os': 1.92, 'Co': 1.67, 'Rh': 1.83, 'Ir': 1.87, 'Ni': 1.62,
                     'Pd': 1.79, 'Pt': 1.83, 'Cu': 1.57, 'Ag': 1.75,
                     'Au': 1.79, 'Zn': 1.53, 'Cd': 1.71, 'Hg': 1.76, 'B': 1.17, 'Al': 1.82, 'Ga': 1.81, 'In': 2,
                     'Tl': 2.08, 'C': 0.91, 'Si': 1.46, 'Ge': 1.52, 'Sn': 1.72,
                     'Pb': 1.81, 'N': 0.75, 'P': 1.23, 'As': 1.33, 'Sb': 1.53, 'Bi': 1.63, 'O': 0.65, 'S': 1.09,
                     'Se': 1.22, 'Te': 1.42, 'Po': 1.53, 'F': 0.57, 'Cl': 0.97, 'Br': 1.12,
                     'I': 1.32, 'At': 1.43, 'La': 2.74, 'Ce': 2.7, 'Pr': 2.67, 'Nd': 2.64, 'Pm': 2.62, 'Eu': 2.56,
                     'Gd': 2.54, 'Tb': 2.51, 'Dy': 2.49, 'Ho': 2.47, 'Er': 2.45,
                     'Tm': 2.42, 'Yb': 2.4, 'Lu': 2.25, 'He': 0.49, 'Ne': 0.51, 'Ar': 0.88, 'Kr': 1.03, 'Xe': 1.24,
                     'Rn': 1.34, 'Fr': 1.8, 'Ra': 1.43, 'Ac': 1.119, 'Th': 0.972, 'Pa': 0.78, 'U': 0.52, 'Np': 0.75,
                     'Pu': 0.887,
                     'Am': 0.982, 'Cm': 0.97, 'Bk': 0.949, 'Cf': 0.934, 'Es': 0.925, 
                     '*':0}

norm_atomic_radius = {atom: atomic_radius[atom]/1 for atom in atomic_radius.keys()}

atomic_volume = {'H': 14.4, 'Li': 13.1, 'Na': 23.7, 'K': 45.46, 'Rb': 55.9, 'Cs': 71.07, 'Be': 5.0, 'Mg': 13.9,
                     'Ca': 29.9, 'Sr': 33.7,
                     'Ba': 39.24, 'Sc': 15.0, 'Y': 19.8, 'Ti': 10.64, 'Zr': 14.1, 'Hf': 13.6, 'V': 8.78, 'Nb': 10.87,
                     'Ta': 10.9, 'Cr': 7.23,
                     'Mo': 9.4, 'W': 9.53, 'Mn': 1.39, 'Tc': 8.5, 'Re': 8.85, 'Fe': 7.1, 'Ru': 8.3, 'Os': 8.49,
                     'Co': 6.7, 'Rh': 8.3, 'Ir': 8.54,
                     'Ni': 6.59, 'Pd': 8.9, 'Pt': 9.1, 'Cu': 7.1, 'Ag': 10.3, 'Au': 10.2, 'Zn': 9.2, 'Cd': 13.1,
                     'Hg': 14.82, 'B': 4.6, 'Al': 10.0,
                     'Ga': 11.8, 'In': 15.7, 'Tl': 7.2, 'C': 4.58, 'Si': 12.1, 'Ge': 13.6, 'Sn': 16.3, 'Pb': 18.17,
                     'N': 17.3, 'P': 17.0, 'As': 13.1,
                     'Sb': 18.23, 'Bi': 21.3, 'O': 14.0, 'S': 15.5, 'Se': 16.45, 'Te': 20.5, 'Po': 22.23, 'F': 17.1,
                     'Cl': 22.7, 'Br': 23.5,
                     'I': 25.74, 'La': 20.73, 'Ce': 20.67, 'Pr': 20.8, 'Nd': 20.6, 'Pm': 28.9, 'Sm': 19.95, 'Eu': 28.9,
                     'Gd': 19.9, 'Tb': 19.2,
                     'Dy': 19.0, 'Ho': 18.7, 'Er': 18.4, 'Tm': 18.1, 'Yb': 24.79, 'Lu': 17.78, 'Ne': 16.7, 'Ar': 28.5,
                     'Kr': 38.9, 'Xe': 37.3,
                     'Rn': 50.5, 'Ra': 45.2, 'Ac': 22.54, 'Th': 19.9, 'Pa': 15, 'U': 12.59, 'Np': 11.62, 'Pu': 12.32,
                     'Am': 17.86, 'Cm': 18.28, '*':0}
norm_atomic_volume = {atom: atomic_volume[atom]/1 for atom in atomic_volume.keys()}                     
chiral_tag_vocab = {Chem.rdchem.ChiralType.values[i]: i for i in range(len(Chem.rdchem.ChiralType.values))}
hybridization_vocab = {Chem.rdchem.HybridizationType.values[i]: i for i in range(len(Chem.rdchem.HybridizationType.values))}

def atom_position(atom):
    """
    Atom position.
    Return 3D position if available, otherwise 2D position is returned.
    """
    mol = atom.GetOwningMol()
    if mol.GetNumConformers() == 0:
        mol.Compute2DCoords()
    conformer = mol.GetConformer()
    pos = conformer.GetAtomPosition(atom.GetIdx())
    return [pos.x, pos.y, pos.z]

def get_stereo_feature(mol, atom):
    stereo_info = Chem.FindMolChiralCenters(mol)
    if stereo_info==[]:
        return [0,0,0]
    else:
        for atom_id, S_R in stereo_info:
            if atom.GetIdx()==atom_id: 
                if S_R=='S': return [1,1,0]
                if S_R=='R': return [1,0,1]
        return [0,0,0]

# bond
bond_type_vocab = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
                   Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
bond_type_vocab = {b: i for i, b in enumerate(bond_type_vocab)}
bond_type_vocab = {Chem.rdchem.BondType.values[i]: i for i in range(len(Chem.rdchem.BondType.values))}
bond_dir_vocab = {Chem.rdchem.BondDir.values[i]: i for i in range(len(Chem.rdchem.BondDir.values))}
bond_stereo_vocab = {Chem.rdchem.BondStereo.values[i]: i for i in range(len(Chem.rdchem.BondStereo.values))}

atom2valence = {1: 1, 5: 3, 6: 4, 7: 3, 8: 2, 9: 1, 14: 4, 15: 5, 16: 6, 17: 1, 35: 1, 53: 7}
bond2valence = [1, 2, 3, 1.5]
bond2id = {"SINGLE": 0, "DOUBLE": 1, "TRIPLE": 2, "AROMATIC": 3}
id2bond = {v: k for k, v in bond2id.items()}
           
def bond_length(bond):
    """Bond length"""
    mol = bond.GetOwningMol()
    if mol.GetNumConformers() == 0:
        mol.Compute2DCoords()
    conformer = mol.GetConformer()
    h = conformer.GetAtomPosition(bond.GetBeginAtomIdx())
    t = conformer.GetAtomPosition(bond.GetEndAtomIdx())
    return [h.Distance(t)]

class Smile2Graph(object):

    def __init__(self, smile, label, with_hydrogen=True, kekulize=False) -> None:
        self.smile = smile
        self.label = label
        self.with_hydrogen = with_hydrogen
        self.kekulze = kekulize

        self.mol = self._init_mol()

        self.empty_mol = Chem.MolFromSmiles("")
        self.dummy_mol = Chem.MolFromSmiles("CC")
        self.dummy_atom = self.dummy_mol.GetAtomWithIdx(0)
        self.dummy_bond = self.dummy_mol.GetBondWithIdx(0)

        self.atoms = [self.mol.GetAtomWithIdx(i) for i in range(self.mol.GetNumAtoms())] 
        self.bonds = [self.mol.GetBondWithIdx(i) for i in range(self.mol.GetNumBonds())] 


    def _init_mol(self):

        mol = Chem.MolFromSmiles(self.smile)

        if mol is None:
            raise ValueError("Invalid SMILES `%s`" % self.smile)

        if self.with_hydrogen:
            mol = Chem.AddHs(mol)

        if self.kekulze:
            Chem.Kekulize(mol)

        return mol

    def get_atom_features(self, atom):

        return [
            atom_vocab[atom.GetSymbol()],
            atom.GetAtomMapNum(), #Gets the atoms map number, returns 0 if not set
            atom.GetAtomicNum(),  #Returns the atomic number.
            chiral_tag_vocab[atom.GetChiralTag()],
            atom.GetDegree(), #Returns the degree of the atom in the molecule.
            atom.GetExplicitValence(), #Returns the explicit valence of the atom.
            atom.GetFormalCharge(), #原子电荷
            hybridization_vocab[atom.GetHybridization()], #returns our hybridization 杂交 全是0 可以舍
            atom.GetImplicitValence(), #返回该原子的隐式价
            int(atom.GetIsAromatic()), #返回是否是芳香的
            int(atom.IsInRing()),
            atom.GetMass(), #返回原子质量
            int(atom.GetNoImplicit()),
            atom.GetNumExplicitHs(),
            atom.GetNumRadicalElectrons(),# 返回此原子的自由基电子数 ---同一个却不同。。。可能是不同---
            atom.GetTotalDegree(),
            atom.GetTotalNumHs(),
            atom.GetTotalValence(),
            norm_atomic_radius[atom.GetSymbol()], #原子半径
            norm_atomic_volume[atom.GetSymbol()], #原子体积
        ] + get_stereo_feature(self.mol, atom)  #+ atom_position(atom)
        
    def get_bond_features(self, bond):

        return [
            bond.GetBeginAtomIdx(), 
            bond_type_vocab[bond.GetBondType()],
            bond.GetEndAtomIdx(),
            bond_dir_vocab[bond.GetBondDir()],
            bond_stereo_vocab[bond.GetStereo()],
            int(bond.GetIsAromatic()),
            int(bond.GetIsConjugated()),     
        ] + bond_length(bond)

    def get_node_feature(self):

        node_feature = []

        for atom in self.atoms:
            # print(atom.GetSymbol())
            node_feature.append(self.get_atom_features(atom))
        
        return torch.tensor(node_feature, dtype=torch.float)

    def get_edge_feature(self):

        edge_feature = []
        for bond in self.bonds:
            edge_feature.append(self.get_bond_features(bond))

        return torch.tensor(edge_feature, dtype=torch.float)

    def get_edge_index(self):

        edge_index = []
        for bond in self.bonds:
            edge_index.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
            edge_index.append([bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()])

        return torch.tensor(edge_index, dtype=torch.long)

    def get_y(self):

        y = self.label

        return torch.tensor([y])


    def to_graph(self):
        """ 
            Data 5 个属性：
            1. x 用于存储每个节点的特征，形状是[num_nodes, num_node_features]
            2. edge_index: 用于存储节点之间的边，形状是 [2, num_edges]
            3. pos: 存储节点的坐标，形状是[num_nodes, num_dimensions]
            4. y: 存储样本标签。如果是每个节点都有标签，那么形状是[num_nodes, *]；如果是整张图只有一个标签，那么形状是[1, *]
            5. edge_attr: 存储边的特征。形状是[num_edges, num_edge_features]
        
        """

        data = Data(
            x=self.get_node_feature(),
            edge_index = self.get_edge_index().t().contiguous(),
            edge_attr = self.get_edge_feature(),
            y = self.get_y()
        )

        return data

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GraphConv(dataset.num_node_features, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.conv3 = GraphConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.lin(x)
        
        return x

if __name__ == "__main__":
    first_param = sys.argv[1]
    second_param = sys.argv[2]
    third_param = sys.argv[3]
    model = None
    if first_param == "Fish":
        with open('F2F.pickle','rb') as f: 
            model = pickle.load(f)
    elif first_param == "Crustaceans":
        with open('C2C.pickle','rb') as f: 
            model = pickle.load(f)
    elif first_param == "Algae":
        with open('A2A.pickle','rb') as f: 
            model = pickle.load(f)
    df = pd.read_csv(second_param)
    result = []
    for i in range(len(df)):
        smi = df.iloc[i,0]
        g = Smile2Graph(smi, label=0)
        data = Data(
            x=g.get_node_feature(),
            edge_index = g.get_edge_index().t().contiguous(),
            edge_attr = g.get_edge_feature(),
            y = g.get_y()
        )
        out = model(data.x, data.edge_index, data.batch)
        probs = torch.softmax(out, dim=1) 
        result.append(probs[0,0].float().item())
    re = pd.DataFrame(result, columns=['Label'])
    re.to_csv(third_param,index=None)
    