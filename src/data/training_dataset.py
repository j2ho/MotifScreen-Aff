"""
Clean training dataset module for MotifScreen-Aff with grouped parameters.
Handles complex training scenarios while maintaining clean architecture.
"""

import os
import sys
import copy
import traceback
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass

import numpy as np
import torch
import dgl
import scipy.spatial

# Local imports
import src.data.types as types
import src.data.utils as myutils
import src.data.kappaidx as kappaidx

logger = logging.getLogger(__name__)

@dataclass
class DataPaths:
    """Data path configuration"""
    datapath: str = 'data'
    keyatomf: str = 'keyatom.def.npz'
    decoyf: str = 'decoys.BL2.npz'
    affinityf: Optional[str] = None

@dataclass
class GraphConfig:
    """Graph construction parameters"""
    # Edge construction
    edgemode: str = 'dist'  # 'dist', 'distT', 'topk'
    edgek: Tuple[int, int] = (8, 16)  # (ligand_k, receptor_k)
    edgedist: Tuple[float, float] = (2.2, 4.5)  # (ligand_dist, receptor_dist)

    # Graph limits
    maxedge: int = 100000
    maxnode: int = 3000

    # Spatial parameters
    ball_radius: float = 8.0  # Receptor neighborhood radius
    firstshell_as_grid: bool = False

@dataclass
class DataProcessing:
    """Data processing configuration"""
    ntype: int = 6  # Number of pharmacophore types
    max_subset: int = 5  # Max ligands per sample
    drop_H: bool = False  # Remove hydrogens
    store_memory: bool = False  # Cache MOL2 data

@dataclass
class DataAugmentation:
    """Data augmentation parameters"""
    randomize: float = 0.5  # Coordinate noise for receptor
    randomize_grid: float = 0.0  # Grid position noise
    pert: bool = False  # Use ligand perturbations

@dataclass
class CrossValidation:
    """Cross-validation settings"""
    load_cross: bool = False  # Load cross-receptor data
    cross_eval_struct: bool = False  # Evaluate cross structures
    cross_grid: float = 0.0  # Probability of using cross grids
    nonnative_struct_weight: float = 0.2  # Weight for non-native structures

@dataclass
class TrainingConfig:
    """Complete training configuration"""
    # Core configs
    paths: DataPaths
    graph: GraphConfig
    processing: DataProcessing
    augmentation: DataAugmentation
    cross_validation: CrossValidation

    # Runtime
    mode: str = 'train'  # 'train', 'valid', 'test'
    debug: bool = False

    @classmethod
    def create_default(cls) -> 'TrainingConfig':
        """Create default configuration"""
        return cls(
            paths=DataPaths(),
            graph=GraphConfig(),
            processing=DataProcessing(),
            augmentation=DataAugmentation(),
            cross_validation=CrossValidation()
        )

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        """Create from dictionary (backward compatibility)"""
        paths = DataPaths(
            datapath=config_dict.get('datapath', 'data'),
            keyatomf=config_dict.get('keyatomf', 'keyatom.def.npz'),
            affinityf=config_dict.get('affinityf', None)
        )

        graph = GraphConfig(
            edgemode=config_dict.get('edgemode', 'dist'),
            edgek=config_dict.get('edgek', (8, 16)),
            edgedist=config_dict.get('edgedist', (2.2, 4.5)),
            maxedge=config_dict.get('maxedge', 100000),
            maxnode=config_dict.get('maxnode', 3000),
            ball_radius=config_dict.get('ball_radius', 8.0),
            firstshell_as_grid=config_dict.get('firstshell_as_grid', False)
        )

        processing = DataProcessing(
            ntype=config_dict.get('ntype', 6),
            max_subset=config_dict.get('max_subset', 5),
            drop_H=config_dict.get('drop_H', False),
            store_memory=config_dict.get('store_memory', False)
        )

        augmentation = DataAugmentation(
            randomize=config_dict.get('randomize', 0.5),
            randomize_grid=config_dict.get('randomize_grid', 0.0),
            pert=config_dict.get('pert', False)
        )

        cross_validation = CrossValidation(
            load_cross=config_dict.get('load_cross', False),
            cross_eval_struct=config_dict.get('cross_eval_struct', False),
            cross_grid=config_dict.get('cross_grid', 0.0),
            nonnative_struct_weight=config_dict.get('nonnative_struct_weight', 0.2)
        )

        return cls(
            paths=paths,
            graph=graph,
            processing=processing,
            augmentation=augmentation,
            cross_validation=cross_validation,
            mode=config_dict.get('mode', 'train'),
            debug=config_dict.get('debug', False)
        )


class MolecularLoader:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.mol2_cache = {}

    def load_keyatoms(self, keyatomf: str) -> Dict[str, List[str]]:
        try:
            data = np.load(keyatomf, allow_pickle=True)
            if 'keyatms' in data:
                return data['keyatms'].item()
            return data
        except Exception as e:
            logger.error(f"Failed to load keyatoms from {keyatomf}: {e}")
            return {}

    def find_keyatomf(self, pname: str, source: str) -> Optional[str]:
        search_paths = [f'{self.config.paths.datapath}/{source}/{self.config.paths.keyatomf}', # biolip, pdbbind
                        f'{self.config.paths.datapath}/{source}/{pname}/batch_mol2s_d3/{pname}.{self.config.paths.keyatomf}',
        ]

        for path in search_paths:
            if os.path.exists(path):
                return path

        logger.error(f"Cannot find keyatomf in {search_paths}")
        return None

    def read_mol2_single(self, mol2_path: str) -> Tuple:
        """Read a single MOL2 file"""
        if not os.path.exists(mol2_path):
            logger.error(f"MOL2 file not found: {mol2_path}")
            return None

        try:
            return myutils.read_mol2(mol2_path, drop_H=self.config.processing.drop_H)
        except Exception as e:
            logger.error(f"Failed to read MOL2 {mol2_path}: {e}")
            return None

    def read_mol2_batch(self, mol2_path: str, tags: List[str] = None) -> Dict:
        """Read batch MOL2 file with caching"""
        if self.config.processing.store_memory and mol2_path in self.mol2_cache:
            return self._extract_from_cache(mol2_path, tags)

        try:
            data = myutils.read_mol2_batch(
                mol2_path,
                drop_H=self.config.processing.drop_H,
                tags_read=tags
            )

            if self.config.processing.store_memory:
                self._cache_mol2_data(mol2_path, data)

            return data
        except Exception as e:
            logger.error(f"Failed to read batch MOL2 {mol2_path}: {e}")
            return None

    def _cache_mol2_data(self, mol2_path: str, data: Tuple):
        """Cache MOL2 data for reuse"""
        elems, qs, bonds, borders, xyz, nneighs, atms, atypes, tags_read = data

        cache_data = {}
        for elem, q, bond, border, coord, nneigh, atm, atype, tag in zip(
            elems, qs, bonds, borders, xyz, nneighs, atms, atypes, tags_read
        ):
            cache_data[tag] = (elem, q, bond, border, coord, nneigh, atm, atype)

        self.mol2_cache[mol2_path] = cache_data
        logger.info(f"Cached MOL2: {mol2_path}, molecules: {len(cache_data)}")

    def _extract_from_cache(self, mol2_path: str, tags: List[str]) -> Tuple:
        """Extract specific tags from cached data"""
        cache_data = self.mol2_cache[mol2_path]

        elems, qs, bonds, borders, xyz, nneighs, atms, atypes, tags_read = [], [], [], [], [], [], [], [], []

        for tag in tags:
            if tag in cache_data:
                elem, q, bond, border, coord, nneigh, atm, atype = cache_data[tag]
                elems.append(elem)
                qs.append(q)
                bonds.append(bond)
                borders.append(border)
                xyz.append(coord)
                nneighs.append(nneigh)
                atms.append(atm)
                atypes.append(atype)
                tags_read.append(tag)

        return elems, qs, bonds, borders, xyz, nneighs, atms, atypes, tags_read


class GraphBuilder:
    """Constructs DGL graphs from molecular data"""

    def __init__(self, config: TrainingConfig):
        self.config = config

    def build_ligand_graph(self, mol_data: Tuple, name: str = "") -> dgl.DGLGraph:
        """Build ligand graph from molecular data"""
        try:
            elems, qs, bonds, borders, xyz, nneighs, atypes = mol_data
            # Build graph connectivity using ligand-specific parameters
            u, v = self._build_edges(xyz, bonds,
                                   mode=self.config.graph.edgemode,
                                   top_k=self.config.graph.edgek[0],  # Ligand K
                                   dcut=self.config.graph.edgedist[0])  # Ligand distance

            graph = dgl.graph((u, v))

            # Add node features
            node_features = self._compute_ligand_node_features(elems, qs, nneighs, xyz, atypes)
            graph.ndata['attr'] = torch.tensor(node_features).float()

            # Add edge features
            edge_features = self._compute_ligand_edge_features(bonds, borders, u, v, xyz)
            graph.edata['attr'] = edge_features

            # Add positional information
            graph.ndata['x'] = torch.tensor(xyz).float()[:, None, :]

            dX = torch.unsqueeze(torch.tensor(xyz)[None, :], 1) - torch.unsqueeze(torch.tensor(xyz)[None, :], 2)
            graph.edata['rel_pos'] = dX[:, u, v].float()[0]

            # Add global features
            global_features = self._compute_global_features(elems, qs, bonds, borders, xyz, atypes)
            setattr(graph, "gdata", torch.tensor(global_features).float())

            # Placeholder for training targets
            graph.ndata['Y'] = torch.zeros(len(elems), 3)

            return graph

        except Exception as e:
            import traceback
            logger.error(f"Failed to build ligand graph for {name}: {e}")
            traceback.print_exc()
            return None

    def build_receptor_graph(self, npz_path: str, grids: np.ndarray, origin: torch.Tensor,
                           gridchain: str = None) -> Tuple[dgl.DGLGraph, np.ndarray, np.ndarray]:
        """Build receptor graph from structure data"""
        try:
            # Load receptor properties
            prop = np.load(npz_path)
            xyz = prop['xyz_rec']
            charges_rec = prop['charge_rec']
            atypes_rec = prop['atypes_rec']
            anames = prop['atmnames']
            aas_rec = prop['aas_rec']
            sasa_rec = prop['sasa_rec']
            bnds = prop['bnds_rec']

            # Process coordinates
            xyz, grids = self._process_coordinates(xyz, grids, origin, gridchain)

            # Apply randomization if configured
            if self.config.augmentation.randomize > 1e-3:
                xyz = self._randomize_coordinates(xyz, self.config.augmentation.randomize)

            # Build combined node features
            node_features = self._compute_receptor_node_features(
                xyz, grids, atypes_rec, anames, aas_rec, sasa_rec, charges_rec
            )

            # Build graph using receptor-specific parameters
            graph = self._build_receptor_graph_structure(
                xyz, grids, node_features, bnds, anames
            )

            if graph is None:
                return None, None, None

            # Apply first shell logic if configured
            if self.config.graph.firstshell_as_grid:
                grids, grid_indices = self._apply_first_shell_logic(graph, xyz, grids)
            else:
                grid_indices = np.where(graph.ndata['attr'][:, 0] == 1)[0]  # aa=unk type for grids

            return graph, grids, grid_indices

        except Exception as e:
            logger.error(f"Failed to build receptor graph from {npz_path}: {e}")
            return None, None, None

    def _build_edges(self, xyz: np.ndarray, bonds: List[List[int]],
                    mode: str, top_k: int, dcut: float) -> Tuple[np.ndarray, np.ndarray]:
        """Build edge connectivity for molecular graph"""
        X = torch.tensor(xyz[None, :])
        dX = torch.unsqueeze(X, 1) - torch.unsqueeze(X, 2)

        u, v, d = self._find_distance_neighbors(dX, mode, top_k, dcut)

        return u, v

    def _find_distance_neighbors(self, dX: torch.Tensor, mode: str = 'dist',
                               top_k: int = 8, dcut: float = 4.5) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Find neighboring atoms based on distance"""
        D = torch.sqrt(torch.sum(dX**2, 3) + 1e-6)

        if mode == 'dist':
            _, u, v = torch.where(D < dcut)
        elif mode == 'distT':
            mask = torch.where(torch.tril(D) < 1e-6, 100.0, 1.0)
            _, u, v = torch.where(mask * D < dcut)
        elif mode == 'topk':
            top_k_var = min(D.shape[1], top_k + 1)
            D_neighbors, E_idx = torch.topk(D, top_k_var, dim=-1, largest=False)
            E_idx = E_idx[:, :, 1:]  # Remove self-connections
            u = torch.arange(E_idx.shape[1])[:, None].repeat(1, E_idx.shape[2]).reshape(-1)
            v = E_idx[0].reshape(-1)

        return u, v, D[0]

    def _compute_ligand_node_features(self, elems: List, qs: List, nneighs: np.ndarray,
                                    xyz: np.ndarray, atypes: List) -> np.ndarray:
        """Compute node features for ligand atoms"""
        features = []

        # Normalize neighbor counts
        ns = np.sum(nneighs, axis=1)[:, None] + 0.01
        ns = np.repeat(ns, 4, axis=1)
        normalized_nneighs = nneighs * (1.0 / ns)
        features.append(normalized_nneighs)

        # SASA and occlusion
        sasa, nsasa, occl = myutils.sasa_from_xyz(xyz, elems)
        features.append(nsasa[:, None])
        features.append(occl[:, None])
        features.append(np.array(qs)[:, None])

        # Element one-hot encoding
        elem_indices = [myutils.ELEMS.index(elem) for elem in elems]
        elem_onehot = np.eye(len(myutils.ELEMS))[elem_indices]
        features.append(elem_onehot)

        return np.concatenate(features, axis=-1)

    def _compute_ligand_edge_features(self, bonds: List, borders: List,
                                    u: np.ndarray, v: np.ndarray, xyz: np.ndarray) -> torch.Tensor:
        """Compute edge features for ligand"""
        # Build bond order matrix
        bond_matrix = np.zeros((len(xyz), len(xyz)), dtype=int)
        bond_indices = np.zeros((len(xyz), len(xyz)), dtype=int)

        for k, (i, j) in enumerate(bonds):
            bond_indices[i, j] = bond_indices[j, i] = k
            bond_matrix[i, j] = bond_matrix[j, i] = 1

        # Extract bond orders for edges
        bond_orders = np.zeros(len(u), dtype=np.int64)
        for k, (i, j) in enumerate(zip(u, v)):
            if bond_matrix[i, j]:
                bond_orders[k] = borders[bond_indices[i, j]]

        # One-hot encode bond orders
        edge_features = torch.eye(5)[bond_orders]  # 0-4 bond types

        # Add topological distance features
        bond_graph = scipy.sparse.csgraph.shortest_path(bond_matrix, directed=False)
        topo_distances = torch.tensor(bond_graph)[u, v]
        edge_features[:, -1] = 1.0 / (topo_distances + 0.00001)  # 1/separation

        return edge_features

    def _compute_global_features(self, elems: List, qs: List, bonds: List,
                               borders: List, xyz: np.ndarray, atypes: List) -> np.ndarray:
        """Compute global molecular features"""
        # Flexible torsion count
        nflextors = 0
        elem_indices = [myutils.ELEMS.index(elem) for elem in elems]

        for i, j in bonds:
            if elem_indices[i] != 1 and elem_indices[j] != 1 and borders[bonds.index([i, j])] > 1:
                nflextors += 1

        # Kappa indices
        kappa = kappaidx.calc_Kappaidx(atypes, bonds, False)
        normalized_kappa = [(kappa[0] - 40.0) / 40.0, (kappa[1] - 15.0) / 15.0, (kappa[2] - 10.0) / 10.0]

        # Molecular inertia
        com = np.mean(xyz, axis=0)
        centered_xyz = xyz - com
        inertia = np.dot(centered_xyz.transpose(), centered_xyz)
        eigvals, _ = np.linalg.eig(inertia)
        principal_values = (np.sqrt(np.sort(eigvals)) - 20.0) / 20.0

        # Atom counts
        natm = (len([e for e in elem_indices if e > 1]) - 25.0) / 25.0
        naro = (len([at for at in atypes if at in ['C.ar', 'C.aro', 'N.ar']]) - 6.0) / 6.0

        # Donor/acceptor counts
        nacc, ndon = -1.0, -1.0
        for i, elem_idx in enumerate(elem_indices):
            if elem_idx not in [3, 4]:  # N, O
                continue
            neighbors = [j for i_bond, j_bond in bonds if (i_bond == i or j_bond == i)]
            has_hydrogen = any(elem_indices[n] == 1 for n in neighbors if n != i)  # H
            if has_hydrogen:
                ndon += 0.2
            else:
                nacc += 0.2

        # Combine features
        nflextors_onehot = np.eye(10)[min(9, nflextors)]
        global_features = np.concatenate([
            nflextors_onehot, normalized_kappa, [nacc, ndon, naro], list(principal_values)
        ])

        return global_features

    def _process_coordinates(self, xyz: np.ndarray, grids: np.ndarray,
                           origin: torch.Tensor, gridchain: str = None) -> Tuple[np.ndarray, np.ndarray]:
        """Process and center coordinates"""
        # Mask chain if specified
        if gridchain is not None:
            # This would need access to residue info - keeping existing logic
            pass

        # Center coordinates
        origin_np = origin.squeeze().numpy()
        xyz_centered = xyz - origin_np
        grids_centered = grids - origin_np

        return xyz_centered, grids_centered

    def _randomize_coordinates(self, xyz: np.ndarray, randomize_factor: float) -> np.ndarray:
        """Apply coordinate randomization"""
        rand_xyz = 2.0 * randomize_factor * (0.5 - np.random.rand(*xyz.shape))
        return xyz + rand_xyz

    def _compute_receptor_node_features(self, xyz: np.ndarray, grids: np.ndarray,
                                      atypes_rec: List, anames: List, aas_rec: List,
                                      sasa_rec: List, charges_rec: List) -> np.ndarray:
        """Compute node features for receptor"""
        ngrids = len(grids)

        # Extend arrays for grid points
        all_aas = np.concatenate([aas_rec, [0] * ngrids])  # unk for grids
        all_xyz = np.concatenate([xyz, grids])

        # Atom types
        atypes = np.array([types.find_gentype2num(at) for at in atypes_rec])
        all_atypes = np.concatenate([atypes, [0] * ngrids])  # null for grids

        # Other features
        all_sasa = np.concatenate([sasa_rec, [0.0] * ngrids])
        all_charges = np.concatenate([charges_rec, [0.0] * ngrids])

        # Distance to origin
        d2o = np.sqrt(np.sum(all_xyz * all_xyz, axis=1))

        # Build feature matrix
        features = []

        # AA one-hot
        aa_onehot = np.eye(types.N_AATYPE)[all_aas]
        features.append(aa_onehot)

        # Atom type one-hot
        atype_onehot = np.eye(max(types.gentype2num.values()) + 1)[all_atypes]
        features.append(atype_onehot)

        # Other features
        features.extend([
            all_sasa[:, None],
            all_charges[:, None],
            d2o[:, None]
        ])

        return np.concatenate(features, axis=-1)

    def _build_receptor_graph_structure(self, xyz: np.ndarray, grids: np.ndarray,
                                      node_features: np.ndarray, bonds: List,
                                      anames: List) -> Optional[dgl.DGLGraph]:
        """Build the actual receptor graph structure"""
        try:
            all_xyz = np.concatenate([xyz, grids])

            # KD-tree neighbor search using configured ball radius
            kd = scipy.spatial.cKDTree(all_xyz)
            kd_grids = scipy.spatial.cKDTree(grids)
            indices = np.concatenate(kd_grids.query_ball_tree(kd, self.config.graph.ball_radius))

            # Get unique atom indices near grids
            selected_indices = list(np.unique(indices).astype(np.int16))

            if len(selected_indices) > self.config.graph.maxnode:
                logger.error(f"Receptor nodes {len(selected_indices)} exceeds max {self.config.graph.maxnode}")
                return None

            # Build adjacency matrix for bonds
            bond_matrix = np.zeros((len(all_xyz), len(all_xyz)))
            index_map = {idx: i for i, idx in enumerate(selected_indices)}

            for i, j in bonds:
                if i in index_map and j in index_map:
                    k, l = index_map[i], index_map[j]
                    bond_matrix[k, l] = bond_matrix[l, k] = 1

            # Self-connections
            # for i in range(len(all_xyz)):
            #     bond_matrix[i, i] = 1

            # Distance-based edges using receptor parameters
            selected_xyz = all_xyz[selected_indices]
            selected_xyz = torch.tensor(selected_xyz).float()
            X = selected_xyz[None, :].clone().detach()
            dX = torch.unsqueeze(X, 1) - torch.unsqueeze(X, 2)

            u, v, D = self._find_distance_neighbors(
                dX,
                mode=self.config.graph.edgemode,
                top_k=self.config.graph.edgek[1],      # Receptor K
                dcut=self.config.graph.edgedist[1]     # Receptor distance
            )

            # Filter edges
            natm = len(xyz)  # Number of real atoms
            within_cutoff = (D < self.config.graph.edgedist[1])
            within_cutoff[:natm, :] = within_cutoff[:, :natm] = 1  # Allow topK for receptor atoms

            valid_edges = within_cutoff[u, v]
            u = u[valid_edges >= 0]
            v = v[valid_edges >= 0]

            # Build graph
            graph = dgl.graph((u, v))

            # Add features
            graph.ndata['attr'] = torch.tensor(node_features[selected_indices]).float()
            graph.ndata['x'] = selected_xyz.clone().detach()[:, None, :]

            # Edge features
            edge_distances = torch.sqrt(torch.sum((selected_xyz[v] - selected_xyz[u]) ** 2, dim=-1) + 1e-6)[:, None]
            edge_features = self._distance_feature(edge_distances, 0.5, 5.0)

            # Bond information
            bond_info = torch.tensor([bond_matrix[v_i, u_i] for u_i, v_i in zip(u, v)]).float()
            edge_features[:, 0] = bond_info

            # Grid neighbor information
            grid_neighbors = ((u >= natm) * (v >= natm)).float()
            edge_features[:, 1] = grid_neighbors

            graph.edata['attr'] = edge_features
            graph.edata['rel_pos'] = dX[:, u, v].float()[0]

            if graph.number_of_edges() > self.config.graph.maxedge:
                logger.error(f"Receptor edges {graph.number_of_edges()} exceeds max {self.config.graph.maxedge}")
                return None

            return graph

        except Exception as e:
            logger.error(f"Error building receptor graph structure: {e}")
            return None

    def _apply_first_shell_logic(self, graph: dgl.DGLGraph, xyz: np.ndarray,
                               grids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply first shell grid logic if configured"""
        if not self.config.graph.firstshell_as_grid:
            grid_indices = np.where(graph.ndata['attr'][:, 0] == 1)[0]
            return grids, grid_indices

        # Existing first shell logic
        recidx = np.where(graph.ndata['attr'][:, 0] == 0)[0]  # aa!=unk type
        grididx = np.where(graph.ndata['attr'][:, 0] == 1)[0]  # aa=unk type

        all_xyz = graph.ndata['x'].squeeze(1).numpy()
        kd_g = scipy.spatial.cKDTree(all_xyz[grididx])
        kd_r = scipy.spatial.cKDTree(all_xyz[recidx])

        firstshell_idx = np.unique(np.concatenate(kd_g.query_ball_tree(kd_r, 4.5)))
        extended_grididx = np.concatenate([grididx, firstshell_idx]).astype(np.int32)

        extended_grids = all_xyz[extended_grididx]

        return extended_grids, extended_grididx

    def _distance_feature(self, distances: torch.Tensor, binsize: float = 0.5,
                         maxd: float = 5.0) -> torch.Tensor:
        """Compute distance-based edge features"""
        d0 = 0.5 * (maxd - binsize)
        m = 5.0 / d0
        feat = 1.0 / (1.0 + torch.exp(-m * (distances - d0)))
        return feat.repeat(1, 3)  # Make 3-dimensional


class TrainingDataSet(torch.utils.data.Dataset):
    """
    Training dataset for MotifScreen-Aff with clean parameter grouping.
    Handles complex training scenarios with organized configuration.
    """

    def __init__(self, targets: List[str], ligands: List = None,
                 config: TrainingConfig = None, **kwargs):
        """
        Initialize training dataset with grouped parameters

        Args:
            targets: List of target identifiers
            ligands: List of ligand specifications
            config: Training configuration (grouped parameters)
            **kwargs: Backward compatibility - will be converted to config
        """
        # Handle configuration
        if config is None:
            # Create config from kwargs for backward compatibility
            config = TrainingConfig.from_dict(kwargs)
        self.config = config

        # Validate inputs
        self.targets = targets
        self.ligands = ligands
        self.decoys = self._load_decoys()
        # Initialize components with grouped config
        self.loader = MolecularLoader(config)
        self.graph_builder = GraphBuilder(config)

        # Load auxiliary data
        self.crossactives = self._load_crossactives() if config.cross_validation.load_cross else {}
        self.affinities = self._load_affinities() if config.paths.affinityf else {}

        logger.info(f"Training dataset initialized with {len(targets)} targets")
        logger.info(f"Graph config: {config.graph}")
        logger.info(f"Processing config: {config.processing}")
        logger.info(f"Augmentation config: {config.augmentation}")

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> Optional[Tuple]:
        """Get dataset item with clean error handling"""
        try:
            return self._get_item_safe(index)
        except Exception as e:
            logger.error(f"Error processing item {index}: {e}")
            return self._get_null_result(self.targets[index])

    def _get_item_safe(self, index: int) -> Optional[Tuple]:
        """Safely get dataset item"""
        target = self.targets[index]
        source = target.split('/')[0]
        pname = target.split('/')[1] # e.g. '1a2b', '4rzu.ADP', 'Q9NQU5'

        # Setup file paths using path config
        receptor_file_paths = self._setup_file_paths(pname, source)
        if not self._validate_files(receptor_file_paths):
            return self._get_null_result(pname)

        # Load key atoms every time ...
        keyatomf = self.loader.find_keyatomf(pname, source)
        if not keyatomf:
            return self._get_null_result(pname)

        keyatoms_dict = self.loader.load_keyatoms(keyatomf)

        # Load pre-computed receptor grid information
        grid_data = self._load_grid_data(receptor_file_paths['gridinfo'])
        if grid_data is None:
            return self._get_null_result(pname)

        grids, cats, mask = grid_data

        # Apply grid augmentation
        if self.config.augmentation.randomize_grid > 1e-3:
            grids = self._apply_grid_randomization(grids)

        # Process ligands if present
        ligand_result = self._process_ligands(target, self.ligands[index], keyatoms_dict)
        if ligand_result is None:
            return self._get_null_result(pname)

        ligand_graphs, native_graph, key_xyz, key_indices, binding_labels, ligand_info = ligand_result

        # Build receptor graph
        origin = torch.tensor(np.mean(grids, axis=0)).float()

        if native_graph is not None:
            native_graph.ndata['x'] = native_graph.ndata['x'] - origin
            key_xyz = key_xyz - origin

        grids = grids - origin.squeeze().numpy()
        gridchain = None
        receptor_graph, processed_grids, grid_indices = self.graph_builder.build_receptor_graph(
            receptor_file_paths['propnpz'], grids, origin, gridchain
        )

        if receptor_graph is None:
            return self._get_null_result(pname)

        # Check graph size limits
        if (receptor_graph.number_of_edges() > self.config.graph.maxedge or
            receptor_graph.number_of_nodes() > self.config.graph.maxnode):
            logger.error(f"Graph too large for {target}: {receptor_graph.number_of_edges()} edges, {receptor_graph.number_of_nodes()} nodes")
            return self._get_null_result(pname)

        # Prepare return info
        info = self._build_info_dict(pname, target, origin, processed_grids,
                                   grid_indices, receptor_file_paths['gridinfo'], ligand_info)

        return (receptor_graph, ligand_graphs, cats, mask, key_xyz,
               key_indices, binding_labels, info)

    def _setup_file_paths(self, pname, source: str) -> Dict[str, str]:
        propnpz = f"{self.config.paths.datapath}/{source}/{pname}/{pname}.prop.npz"
        gridinfo = f"{self.config.paths.datapath}/{source}/{pname}/{pname}.grid.npz"
        parentpath = '/'.join(gridinfo.split('/')[:-1]) + '/'

        return {
            'propnpz': propnpz,
            'gridinfo': gridinfo,
            'parentpath': parentpath
        }

    def _validate_files(self, file_paths: Dict[str, str]) -> bool:
        """Validate that required files exist"""
        required_files = ['gridinfo', 'propnpz']
        for file_key in required_files:
            if not os.path.exists(file_paths[file_key]):
                logger.error(f"Required file missing: {file_paths[file_key]}")
                return False
        return True

    def _load_grid_data(self, gridinfo_path: str) -> Optional[Tuple]:
        """Load grid information from file"""
        try:
            sample = np.load(gridinfo_path, allow_pickle=True)
            grids = sample['xyz']
            # Load labels and masks
            cats, mask = None, None
            if 'labels' in sample or 'label' in sample:
                cats = sample.get('labels', sample.get('label'))
                if len(cats) > 0:
                    if cats.shape[1] > self.config.processing.ntype:
                        cats = cats[:, :self.config.processing.ntype]
                    mask = np.sum(cats > 0, axis=1)
                    cats = torch.tensor(cats).float()
                    mask = torch.tensor(mask).float()

            return grids, cats, mask

        except Exception as e:
            logger.error(f"Error loading grid data from {gridinfo_path}: {e}")
            return None

    def _apply_grid_randomization(self, grids: np.ndarray) -> np.ndarray:
        """Apply grid randomization using augmentation config"""
        rand_xyz = 2.0 * self.config.augmentation.randomize_grid * (0.5 - np.random.rand(len(grids), 3))
        return grids + rand_xyz

    def _process_ligands(self, target: str, ligands: List,
                        keyatoms_dict: Dict) -> Optional[Tuple]:
        """Process ligand data"""
        try:
            # Parse ligand specification
            ligands_parsed, mol2_type, data_type = self._parse_ligands(target, ligands)

            if ligands_parsed is None:
                return None

            # Load ligand structures
            if mol2_type == 'batch':
                ligand_data = self._load_batch_ligands(target, ligands_parsed, keyatoms_dict)
            elif mol2_type == 'single':
                ligand_data = self._load_single_ligands(target, ligands_parsed, keyatoms_dict)

            if ligand_data is None:
                return None

            ligand_graphs, native_graphs, key_indices_list, atoms_list, tags_read = ligand_data
            if len(ligand_graphs) == 0:
                logger.error(f"No valid ligand graphs found for {target}")
                return None
            # Determine active ligands
            actives = [ligands_parsed[0]] #if self.config.mode == 'train' else []
            binding_labels = [1 if tag in actives else 0 for tag in tags_read]

            # Extract key coordinates from native structure
            key_xyz = torch.zeros((4, 3))
            if actives and actives[0] in tags_read:
                active_idx = tags_read.index(actives[0])
                native_graph = native_graphs[active_idx] if native_graphs else None
                if not isinstance(native_graph, list):
                    if native_graph is not None:
                        key_xyz = native_graph.ndata['x'][key_indices_list[active_idx]]

            ligand_info = {
                'nK': [len(idx) for idx in key_indices_list],
                'ligands': tags_read,
                'atms': atoms_list,
            }

            return (ligand_graphs, native_graph, key_xyz, key_indices_list,
                   binding_labels, ligand_info)

        except Exception as e:
            logger.error(f"Error processing ligands for {target}: {e}")
            traceback.print_exc()
            return None

    def _parse_ligands(self, target: str, ligands: List) -> Tuple:
        """Parse ligand specifications with cross-validation logic"""
        source = target.split('/')[0]
        pname = target.split('/')[1]
        # Determine data type
        if source in ['docked']:
            data_type = 'model'
        elif source in ['biolip','pdbbind']:
            data_type = 'structure'
        elif source in ['chembl']:
            data_type = 'activity'

        (active_ligand, mol2_type) = ligands[0], ligands[1] if len(ligands) > 1 else 'single'
        if mol2_type == 'single':
            active = [pname]
            decoys = self.decoys.get(pname, [])
        elif mol2_type == 'batch':
            mol2_file = f"{self.config.paths.datapath}/{source}/{pname}/batch_mol2s_d3/{active_ligand}_b.mol2"
            if os.path.exists(mol2_file):
                active_and_decoys = myutils.read_mol2_batch(mol2_file, tag_only=True)[-1]
                active = [active_and_decoys[0]]  # First is active
                decoys = active_and_decoys[1:]  # Skip first (active)
            else:
                return None, None, None
        else:
            raise ValueError(f"Unknown mol2_type: {mol2_type}")

        # Limit decoy count using processing config
        if len(decoys) > self.config.processing.max_subset - 1:
            decoys = list(np.random.choice(decoys, self.config.processing.max_subset - len(active), replace=False))

        final_ligands = active + decoys

        return final_ligands, mol2_type, data_type

    def _load_batch_ligands(self, target: str, ligands: List[str],
                          keyatoms_dict: Dict) -> Optional[Tuple]:
        """Load ligands from batch MOL2 file"""
        source = target.split('/')[0]
        pname = target.split('/')[1]
        active_ligand = ligands[0]  # First ligand is considered active
        mol2_file = f"{self.config.paths.datapath}/{source}/{pname}/batch_mol2s_d3/{active_ligand}_b.mol2"
        try:
            mol_data = self.loader.read_mol2_batch(mol2_file, ligands)
            if mol_data is None:
                return None

            elems, qs, bonds, borders, xyz, nneighs, atms, atypes, tags_read = mol_data

            ligand_graphs = []
            key_indices_list = []
            atoms_list = []
            tags_processed = []

            for elem, q, bond, border, coord, nneigh, atm, atype, tag in zip(
                elems, qs, bonds, borders, xyz, nneighs, atms, atypes, tags_read
            ):
                try:
                    # Build ligand graph
                    mol_tuple = (elem, q, bond, border, coord, nneigh, atype)
                    graph = self.graph_builder.build_ligand_graph(mol_tuple, name=tag)

                    if graph is None:
                        continue

                    # Center coordinates
                    com = torch.mean(graph.ndata['x'], axis=0).float()
                    graph.ndata['x'] = (graph.ndata['x'] - com).float()

                    # Identify key atoms
                    if self.config.processing.drop_H:
                        filtered_atoms = [atom for atom, element in zip(atm, elem) if element != 'H']
                    else:
                        filtered_atoms = atm

                    key_indices = self._identify_key_atoms(tag, filtered_atoms, keyatoms_dict)

                    if not key_indices:
                        logger.warning(f"No key atoms found for {tag}")
                        continue

                    ligand_graphs.append(graph)
                    key_indices_list.append(key_indices)
                    atoms_list.append(filtered_atoms)
                    tags_processed.append(tag)

                except Exception as e:
                    logger.error(f"Error processing ligand {tag}: {e}")
                    continue

            # Native graphs are same as ligand graphs for batch processing
            native_graphs = ligand_graphs

            return ligand_graphs, native_graphs, key_indices_list, atoms_list, tags_processed

        except Exception as e:
            logger.error(f"Error loading batch ligands from {mol2_file}: {e}")
            return None

    def _load_single_ligands(self, target: str, ligands: List[str], keyatoms_dict: Dict) -> Optional[Tuple]:
        """Load ligands from individual MOL2 files"""
        source = target.split('/')[0]
        pname = target.split('/')[1]
        try:
            ligand_graphs = []
            native_graphs = []
            key_indices_list = []
            atoms_list = []
            tags_read = []

            for i, ligand in enumerate(ligands):
                if i == 0:
                    mol2_path = f"{self.config.paths.datapath}/{source}/{pname}/{pname}.ligand.mol2"
                    lig_pname = pname
                else:
                    ligand = ligand.split('/')
                    lig_source = ligand[0].split('.')[-1]
                    if lig_source == 'ligand':
                        lig_source = 'pdbbind'
                    lig_pname = ligand[1]
                    mol2_path = f"{self.config.paths.datapath}/{lig_source}/{lig_pname}/{lig_pname}.ligand.mol2"
                try:
                    # Construct MOL2 path
                    if mol2_path.endswith('.mol2'):
                        mol2_file = mol2_path
                        conf_mol2 = mol2_path[:-5] + '.conformers.mol2'
                    else:
                        mol2_file = os.path.join(mol2_path, f'{ligand}.ligand.mol2')
                        conf_mol2 = os.path.join(mol2_path, f'{ligand}.conformers.mol2')

                    # Clean ligand name
                    clean_ligand = lig_pname #ligand.replace('GridNet.ligand', '').replace('GridNet.', '')

                    # Read MOL2 data
                    mol_data = self.loader.read_mol2_single(mol2_file)
                    if mol_data is None:
                        logger.warning(f"Failed to read {mol2_file}")
                        continue

                    # Build graphs
                    ligand_graph = self.graph_builder.build_ligand_graph(mol_data, name=clean_ligand)
                    if ligand_graph is None:
                        continue

                    # Center coordinates
                    com = torch.mean(ligand_graph.ndata['x'], axis=0).float()
                    ligand_graph.ndata['x'] = (ligand_graph.ndata['x'] - com).float()

                    # Native graph (could be conformer)
                    native_graph = copy.deepcopy(ligand_graph)
                    if (os.path.exists(conf_mol2) and self.config.augmentation.pert and
                        not conf_mol2.endswith(mol2_file)):
                        # Load conformer coordinates
                        try:
                            conf_xyz, _ = myutils.read_mol2s_xyzonly(conf_mol2)
                            if conf_xyz:
                                conf_idx = min(np.random.randint(len(conf_xyz)), len(conf_xyz) - 1)
                                native_graph.ndata['x'] = torch.tensor(conf_xyz[conf_idx]).float()[:, None, :]
                        except:
                            pass

                    # Identify key atoms
                    atoms = mol_data[6]  # atom names
                    key_indices = self._identify_key_atoms(clean_ligand, atoms, keyatoms_dict)
                    if not key_indices:
                        continue

                    ligand_graphs.append(ligand_graph)
                    native_graphs.append(native_graph)
                    key_indices_list.append(key_indices)
                    atoms_list.append(atoms)
                    tags_read.append(clean_ligand)

                except Exception as e:
                    logger.error(f"Error processing single ligand {ligand}: {e}")
                    continue
            return ligand_graphs, native_graphs, key_indices_list, atoms_list, tags_read

        except Exception as e:
            logger.error(f"Error loading single ligands: {e}")
            return None

    def _identify_key_atoms(self, target: str, atoms: List[str], keyatoms_dict: Dict) -> List[int]:
        """Identify key atom indices for a target"""
        if target not in keyatoms_dict:
            return []

        key_indices = [atoms.index(atom) for atom in keyatoms_dict[target] if atom in atoms]

        if len(key_indices) > 10:
            key_indices = list(np.random.choice(key_indices, 10, replace=False))

        return key_indices

    def _build_info_dict(self, pname: str, target: str, origin: torch.Tensor,
                        grids: np.ndarray, grid_indices: np.ndarray, gridinfo: str,
                        ligand_info: Dict) -> Dict:
        """Build information dictionary for return"""
        source = target.split('/')[0]
        if source in ['docked', 'biolip', 'pdbbind']:
            eval_struct = 1
        elif source in ['chembl']:
            eval_struct = 0
        info = {
            'pname': pname,
            'name': target,
            'com': origin,
            'grididx': grid_indices,
            'grid': grids,
            'gridinfo': gridinfo,
            'source': source,
            'eval_struct': self.config.cross_validation.nonnative_struct_weight if 'model' in target else eval_struct
        }

        # Add ligand-specific info
        info.update(ligand_info)

        return info

    def _load_decoys(self) -> Dict:
        """Load decoy databases"""
        decoys = {}
        file_path = os.path.join(self.config.paths.datapath, self.config.paths.decoyf)
        if os.path.exists(file_path):
            logger.info(f"Loading decoy NPZ: {file_path}")
            decoys = np.load(file_path, allow_pickle=True)['decoys'].item()
        return decoys

    def _load_crossactives(self) -> Dict:
        """Load cross-receptor active compounds"""
        mypath = os.path.dirname(os.path.abspath(__file__))
        cross_file = os.path.join(mypath, '../../data/crossreceptor.filtered.npz')

        if os.path.exists(cross_file):
            return np.load(cross_file, allow_pickle=True)['crossrec'].item()
        return {}

    def _load_affinities(self) -> Dict:
        """Load affinity data"""
        if not self.config.paths.affinityf:
            return {}

        try:
            return parse_affinity(self.config.paths.affinityf)
        except Exception as e:
            logger.error(f"Error loading affinity file {self.config.paths.affinityf}: {e}")
            return {}

    def _get_null_result(self, pname: str) -> Tuple:
        """Return null result for failed processing"""
        info = {'pname': pname}
        return (None, None, None, None, None, None, None, info)


# Keep the same collate function
def collate(samples):
    """Collate function for DataLoader"""
    # Filter valid samples
    valid_samples = [s for s in samples if s is not None and s[0] is not None]
    if not valid_samples:
        return None
    # Extract components
    receptor_graphs = [s[0] for s in valid_samples]
    ligand_graphs = [s[1] for s in valid_samples]
    cats = [s[2] for s in valid_samples]
    masks = [s[3] for s in valid_samples]
    key_xyz = [s[4] for s in valid_samples]
    key_indices = [s[5] for s in valid_samples]
    binding_labels = [s[6] for s in valid_samples]
    info_dicts = [s[-1] for s in valid_samples]

    # Batch receptor graphs
    batched_receptors = dgl.batch(receptor_graphs)

    # Handle categorical data
    if None in cats:
        batched_cats = None
        batched_masks = None
    else:
        batched_cats = torch.stack(cats, dim=0).squeeze()
        if len(batched_cats.shape) == 2:
            batched_cats = batched_cats[None, :]

        batched_masks = torch.stack(masks, dim=0).float().squeeze()
        if len(batched_masks.shape) == 1:
            batched_masks = batched_masks[None, :]

    # Combine info dictionaries
    combined_info = {}
    for key in info_dicts[0]:
        combined_info[key] = [info_dict[key] for info_dict in info_dicts]

    # Convert grid info to tensor
    combined_info['grid'] = torch.tensor(np.array(combined_info['grid']))

    # Handle grid indices for batched graph
    batched_grid_indices = []
    node_offset = 0
    for num_nodes, grid_idx in zip(batched_receptors.batch_num_nodes(), combined_info['grididx']):
        batched_grid_indices.append(torch.tensor(grid_idx, dtype=int) + node_offset)
        node_offset += num_nodes
    combined_info['grididx'] = torch.cat(batched_grid_indices, dim=0)

    # Handle ligand data
    if ligand_graphs[0]:
        batched_ligands = dgl.batch(ligand_graphs[0])

        # Batch global ligand data
        global_data = torch.stack([g.gdata for g in ligand_graphs[0]])
        setattr(batched_ligands, "gdata", global_data)

        # Handle key coordinates and indices
        try:
            batched_key_xyz = key_xyz[0].squeeze()[None, :, :]
            batched_key_indices = [torch.eye(n)[idx] for n, idx in zip(batched_ligands.batch_num_nodes(), key_indices[0])]
            batched_binding_labels = torch.tensor(binding_labels[0], dtype=float)

            combined_info['nK'] = torch.tensor(combined_info['nK'][0])
        except:
            batched_ligands = None
            batched_key_xyz = torch.tensor(0.0)
            batched_key_indices = torch.tensor(0.0)
            batched_binding_labels = torch.tensor([])
    else:
        batched_ligands = None
        batched_key_xyz = torch.tensor(0.0)
        batched_key_indices = torch.tensor(0.0)
        batched_binding_labels = torch.tensor([])

    return (batched_receptors, batched_ligands, batched_cats, batched_masks,
           batched_key_xyz, batched_key_indices, batched_binding_labels, combined_info)


# Backward compatibility functions
def parse_affinity(affinity_file: str) -> Dict[str, float]:
    """Parse affinity data file"""
    data_dict = {}
    try:
        import csv

        with open(affinity_file, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                key_id = f"{row['pdb_id']}.{row['ligand_id']}"
                key_pdbonly = f"{row['pdb_id']}"
                if key_id not in data_dict:
                    value = float(row['pAff_selected'])
                    data_dict[key_id] = value
                    if key_pdbonly not in data_dict:
                        data_dict[key_pdbonly] = value
    except Exception as e:
        logger.error(f"Error parsing affinity file {affinity_file}: {e}")
    return data_dict


def find_aff(active_csv: str, ligand_id: str) -> Optional[float]:
    """Find affinity value for ligand"""
    try:
        with open(active_csv, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 3 and parts[1] == ligand_id:
                    aff_nM = float(parts[2])
                    if aff_nM <= 0:
                        return None
                    aff_M = aff_nM / 1e9
                    return -np.log10(aff_M)  # pIC50
    except Exception as e:
        logger.error(f"Error finding affinity for {ligand_id}: {e}")
    return None


# Backward compatibility alias
DataSet = TrainingDataSet
