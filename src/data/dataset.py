# src/data/training_dataset.py

import os
import copy
import traceback
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
# from dataclasses import dataclass # No longer needed here if imported from config

import numpy as np
import torch
import dgl
import scipy.spatial

# Local imports
import src.data.types as types
import src.data.utils as myutils
import src.data.kappaidx as kappaidx
from configs.config_loader import Config, DataPathsConfig, GraphParamsConfig, DataProcessingConfig, DataAugmentationConfig, CrossValidationConfig # Import your canonical config classes

logger = logging.getLogger(__name__)

class MolecularLoader:
    # Change config type hint to relevant sub-config or general Config
    def __init__(self, config_paths: DataPathsConfig, config_processing: DataProcessingConfig, config_augmentation: DataAugmentationConfig):
        self.config_paths = config_paths
        self.config_processing = config_processing
        self.config_augmentation = config_augmentation # Augmentation can be passed down if needed in loader
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
        # Access config via self.config_paths
        search_paths = [f'{self.config_paths.datapath}/{source}/{self.config_paths.keyatomf}',
                        f'{self.config_paths.datapath}/{source}/{pname}/batch_mol2s_d3/{pname}.{self.config_paths.keyatomf}',
        ]

        for path in search_paths:
            if os.path.exists(path):
                return path

        logger.error(f"Cannot find keyatomf in {search_paths}")
        return None

    def read_mol2_single(self, mol2_path: str) -> Tuple:
        # Access config via self.config_processing
        if not os.path.exists(mol2_path):
            logger.error(f"MOL2 file not found: {mol2_path}")
            return None
        try:
            return myutils.read_mol2(mol2_path, drop_H=self.config_processing.drop_H)
        except Exception as e:
            logger.error(f"Failed to read MOL2 {mol2_path}: {e}")
            return None

    def read_mol2_batch(self, mol2_path: str, tags: List[str] = None) -> Dict:
        # Access config via self.config_processing
        if self.config_processing.store_memory and mol2_path in self.mol2_cache:
            return self._extract_from_cache(mol2_path, tags)
        try:
            data = myutils.read_mol2_batch(
                mol2_path,
                drop_H=self.config_processing.drop_H,
                tags_read=tags
            )
            if self.config_processing.store_memory:
                self._cache_mol2_data(mol2_path, data)
            return data
        except Exception as e:
            logger.error(f"Failed to read batch MOL2 {mol2_path}: {e}")
            return None

    def _cache_mol2_data(self, mol2_path: str, data: Tuple):
        elems, qs, bonds, borders, xyz, nneighs, atms, atypes, tags_read = data
        cache_data = {}
        for elem, q, bond, border, coord, nneigh, atm, atype, tag in zip(
            elems, qs, bonds, borders, xyz, nneighs, atms, atypes, tags_read
        ):
            cache_data[tag] = (elem, q, bond, border, coord, nneigh, atm, atype)
        self.mol2_cache[mol2_path] = cache_data
        logger.info(f"Cached MOL2: {mol2_path}, molecules: {len(cache_data)}")

    def _extract_from_cache(self, mol2_path: str, tags: List[str]) -> Tuple:
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
    def __init__(self, config_graph: GraphParamsConfig, config_augmentation: DataAugmentationConfig, config_processing: DataProcessingConfig):
        self.config_graph = config_graph
        self.config_augmentation = config_augmentation
        self.config_processing = config_processing

    def build_ligand_graph(self, mol_data: Tuple, name: str = "") -> dgl.DGLGraph:
        try:
            elems, qs, bonds, borders, xyz, nneighs, atypes = mol_data
            u, v = self._build_edges(xyz, bonds,
                                   mode=self.config_graph.edgemode,
                                   top_k=self.config_graph.edgek[0],
                                   dcut=self.config_graph.edgedist[0])
            graph = dgl.graph((u, v))
            node_features = self._compute_ligand_node_features(elems, qs, nneighs, xyz, atypes)
            graph.ndata['attr'] = torch.tensor(node_features).float()
            edge_features = self._compute_ligand_edge_features(bonds, borders, u, v, xyz)
            graph.edata['attr'] = edge_features
            graph.ndata['x'] = torch.tensor(xyz).float()[:, None, :]
            dX = torch.unsqueeze(torch.tensor(xyz)[None, :], 1) - torch.unsqueeze(torch.tensor(xyz)[None, :], 2)
            graph.edata['rel_pos'] = dX[:, u, v].float()[0]
            global_features = self._compute_global_features(elems, qs, bonds, borders, xyz, atypes)
            setattr(graph, "gdata", torch.tensor(global_features).float())
            graph.ndata['Y'] = torch.zeros(len(elems), 3)
            return graph
        except Exception as e:
            import traceback
            logger.error(f"Failed to build ligand graph for {name}: {e}")
            traceback.print_exc()
            return None

    def build_receptor_graph(self, npz_path: str, grids: np.ndarray, origin: torch.Tensor,
                           gridchain: str = None) -> Tuple[dgl.DGLGraph, np.ndarray, np.ndarray]:
        try:
            prop = np.load(npz_path)
            xyz = prop['xyz_rec']
            charges_rec = prop['charge_rec']
            atypes_rec = prop['atypes_rec']
            anames = prop['atmnames']
            aas_rec = prop['aas_rec']
            sasa_rec = prop['sasa_rec']
            bnds = prop['bnds_rec']

            xyz, grids = self._process_coordinates(xyz, grids, origin, gridchain)

            if self.config_augmentation.randomize > 1e-3:
                xyz = self._randomize_coordinates(xyz, self.config_augmentation.randomize)

            node_features = self._compute_receptor_node_features(
                xyz, grids, atypes_rec, anames, aas_rec, sasa_rec, charges_rec
            )

            graph = self._build_receptor_graph_structure(
                xyz, grids, node_features, bnds, anames
            )

            if graph is None:
                return None, None, None

            if self.config_graph.firstshell_as_grid:
                grids, grid_indices = self._apply_first_shell_logic(graph, xyz, grids)
            else:
                grid_indices = np.where(graph.ndata['attr'][:, 0] == 1)[0]

            return graph, grids, grid_indices

        except Exception as e:
            logger.error(f"Failed to build receptor graph from {npz_path}: {e}")
            return None, None, None

    def _build_edges(self, xyz: np.ndarray, bonds: List[List[int]],
                    mode: str, top_k: int, dcut: float) -> Tuple[np.ndarray, np.ndarray]:
        X = torch.tensor(xyz[None, :])
        dX = torch.unsqueeze(X, 1) - torch.unsqueeze(X, 2)
        u, v, d = self._find_distance_neighbors(dX, mode, top_k, dcut)
        return u, v

    def _find_distance_neighbors(self, dX: torch.Tensor, mode: str = 'dist',
                               top_k: int = 8, dcut: float = 4.5) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        D = torch.sqrt(torch.sum(dX**2, 3) + 1e-6)
        if mode == 'dist':
            _, u, v = torch.where(D < dcut)
        elif mode == 'distT':
            mask = torch.where(torch.tril(D) < 1e-6, 100.0, 1.0)
            _, u, v = torch.where(mask * D < dcut)
        elif mode == 'topk':
            top_k_var = min(D.shape[1], top_k + 1)
            D_neighbors, E_idx = torch.topk(D, top_k_var, dim=-1, largest=False)
            E_idx = E_idx[:, :, 1:]
            u = torch.arange(E_idx.shape[1])[:, None].repeat(1, E_idx.shape[2]).reshape(-1)
            v = E_idx[0].reshape(-1)
        return u, v, D[0]

    def _compute_ligand_node_features(self, elems: List, qs: List, nneighs: np.ndarray,
                                    xyz: np.ndarray, atypes: List) -> np.ndarray:
        features = []
        ns = np.sum(nneighs, axis=1)[:, None] + 0.01
        ns = np.repeat(ns, 4, axis=1)
        normalized_nneighs = nneighs * (1.0 / ns)
        features.append(normalized_nneighs)
        sasa, nsasa, occl = myutils.sasa_from_xyz(xyz, elems)
        features.append(nsasa[:, None])
        features.append(occl[:, None])
        features.append(np.array(qs)[:, None])
        elem_indices = [myutils.ELEMS.index(elem) for elem in elems]
        elem_onehot = np.eye(len(myutils.ELEMS))[elem_indices]
        features.append(elem_onehot)
        return np.concatenate(features, axis=-1)

    def _compute_ligand_edge_features(self, bonds: List, borders: List,
                                    u: np.ndarray, v: np.ndarray, xyz: np.ndarray) -> torch.Tensor:
        bond_matrix = np.zeros((len(xyz), len(xyz)), dtype=int)
        bond_indices = np.zeros((len(xyz), len(xyz)), dtype=int)
        for k, (i, j) in enumerate(bonds):
            bond_indices[i, j] = bond_indices[j, i] = k
            bond_matrix[i, j] = bond_matrix[j, i] = 1
        bond_orders = np.zeros(len(u), dtype=np.int64)
        for k, (i, j) in enumerate(zip(u, v)):
            if bond_matrix[i, j]:
                bond_orders[k] = borders[bond_indices[i, j]]
        edge_features = torch.eye(5)[bond_orders]
        bond_graph = scipy.sparse.csgraph.shortest_path(bond_matrix, directed=False)
        topo_distances = torch.tensor(bond_graph)[u, v]
        edge_features[:, -1] = 1.0 / (topo_distances + 0.00001)
        return edge_features

    def _compute_global_features(self, elems: List, qs: List, bonds: List,
                               borders: List, xyz: np.ndarray, atypes: List) -> np.ndarray:
        nflextors = 0
        elem_indices = [myutils.ELEMS.index(elem) for elem in elems]
        for i, j in bonds:
            if elem_indices[i] != 1 and elem_indices[j] != 1 and borders[bonds.index([i, j])] > 1:
                nflextors += 1
        kappa = kappaidx.calc_Kappaidx(atypes, bonds, False)
        normalized_kappa = [(kappa[0] - 40.0) / 40.0, (kappa[1] - 15.0) / 15.0, (kappa[2] - 10.0) / 10.0]
        com = np.mean(xyz, axis=0)
        centered_xyz = xyz - com
        inertia = np.dot(centered_xyz.transpose(), centered_xyz)
        eigvals, _ = np.linalg.eig(inertia)
        principal_values = (np.sqrt(np.sort(eigvals)) - 20.0) / 20.0
        natm = (len([e for e in elem_indices if e > 1]) - 25.0) / 25.0
        naro = (len([at for at in atypes if at in ['C.ar', 'C.aro', 'N.ar']]) - 6.0) / 6.0
        nacc, ndon = -1.0, -1.0
        for i, elem_idx in enumerate(elem_indices):
            if elem_idx not in [3, 4]:
                continue
            neighbors = [j for i_bond, j_bond in bonds if (i_bond == i or j_bond == i)]
            has_hydrogen = any(elem_indices[n] == 1 for n in neighbors if n != i)
            if has_hydrogen:
                ndon += 0.2
            else:
                nacc += 0.2
        nflextors_onehot = np.eye(10)[min(9, nflextors)]
        global_features = np.concatenate([
            nflextors_onehot, normalized_kappa, [nacc, ndon, naro], list(principal_values)
        ])
        return global_features

    def _process_coordinates(self, xyz: np.ndarray, grids: np.ndarray,
                           origin: torch.Tensor, gridchain: str = None) -> Tuple[np.ndarray, np.ndarray]:
        if gridchain is not None:
            pass
        origin_np = origin.squeeze().numpy()
        xyz_centered = xyz - origin_np
        grids_centered = grids - origin_np
        return xyz_centered, grids_centered

    def _randomize_coordinates(self, xyz: np.ndarray, randomize_factor: float) -> np.ndarray:
        rand_xyz = 2.0 * randomize_factor * (0.5 - np.random.rand(*xyz.shape))
        return xyz + rand_xyz

    def _compute_receptor_node_features(self, xyz: np.ndarray, grids: np.ndarray,
                                      atypes_rec: List, anames: List, aas_rec: List,
                                      sasa_rec: List, charges_rec: List) -> np.ndarray:
        ngrids = len(grids)
        all_aas = np.concatenate([aas_rec, [0] * ngrids])
        all_xyz = np.concatenate([xyz, grids])
        atypes = np.array([types.find_gentype2num(at) for at in atypes_rec])
        all_atypes = np.concatenate([atypes, [0] * ngrids])
        all_sasa = np.concatenate([sasa_rec, [0.0] * ngrids])
        all_charges = np.concatenate([charges_rec, [0.0] * ngrids])
        d2o = np.sqrt(np.sum(all_xyz * all_xyz, axis=1))
        features = []
        aa_onehot = np.eye(types.N_AATYPE)[all_aas]
        features.append(aa_onehot)
        atype_onehot = np.eye(max(types.gentype2num.values()) + 1)[all_atypes]
        features.append(atype_onehot)
        features.extend([
            all_sasa[:, None],
            all_charges[:, None],
            d2o[:, None]
        ])
        return np.concatenate(features, axis=-1)

    def _build_receptor_graph_structure(self, xyz: np.ndarray, grids: np.ndarray,
                                      node_features: np.ndarray, bonds: List,
                                      anames: List) -> Optional[dgl.DGLGraph]:
        try:
            all_xyz = np.concatenate([xyz, grids])
            kd = scipy.spatial.cKDTree(all_xyz)
            kd_grids = scipy.spatial.cKDTree(grids)
            indices = np.concatenate(kd_grids.query_ball_tree(kd, self.config_graph.ball_radius))
            selected_indices = list(np.unique(indices).astype(np.int16))
            if len(selected_indices) > self.config_graph.maxnode:
                logger.error(f"Receptor nodes {len(selected_indices)} exceeds max {self.config_graph.maxnode}")
                return None
            bond_matrix = np.zeros((len(all_xyz), len(all_xyz)))
            index_map = {idx: i for i, idx in enumerate(selected_indices)}
            for i, j in bonds:
                if i in index_map and j in index_map:
                    k, l = index_map[i], index_map[j]
                    bond_matrix[k, l] = bond_matrix[l, k] = 1
            # Uncomment if want to ensure self-loops
            # for i in range(len(all_xyz)):
            #     bond_matrix[i, i] = 1
            selected_xyz = all_xyz[selected_indices]
            selected_xyz = torch.tensor(selected_xyz).float()
            X = selected_xyz[None, :].clone().detach()
            dX = torch.unsqueeze(X, 1) - torch.unsqueeze(X, 2)
            u, v, D = self._find_distance_neighbors(
                dX,
                mode=self.config_graph.edgemode,
                top_k=self.config_graph.edgek[1],
                dcut=self.config_graph.edgedist[1]
            )
            natm = len(xyz)
            within_cutoff = (D < self.config_graph.edgedist[1])
            within_cutoff[:natm, :] = within_cutoff[:, :natm] = 1
            valid_edges = within_cutoff[u, v]
            u = u[valid_edges >= 0]
            v = v[valid_edges >= 0]
            graph = dgl.graph((u, v))
            graph.ndata['attr'] = torch.tensor(node_features[selected_indices]).float()
            graph.ndata['x'] = selected_xyz.clone().detach()[:, None, :]
            edge_distances = torch.sqrt(torch.sum((selected_xyz[v] - selected_xyz[u]) ** 2, dim=-1) + 1e-6)[:, None]
            edge_features = self._distance_feature(edge_distances, 0.5, 5.0)
            bond_info = torch.tensor([bond_matrix[v_i, u_i] for u_i, v_i in zip(u, v)]).float()
            edge_features[:, 0] = bond_info
            grid_neighbors = ((u >= natm) * (v >= natm)).float()
            edge_features[:, 1] = grid_neighbors
            graph.edata['attr'] = edge_features
            graph.edata['rel_pos'] = dX[:, u, v].float()[0]
            if graph.number_of_edges() > self.config_graph.maxedge:
                logger.error(f"Receptor edges {graph.number_of_edges()} exceeds max {self.config_graph.maxedge}")
                return None
            return graph
        except Exception as e:
            logger.error(f"Error building receptor graph structure: {e}")
            return None

    def _apply_first_shell_logic(self, graph: dgl.DGLGraph, xyz: np.ndarray,
                               grids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self.config_graph.firstshell_as_grid:
            grid_indices = np.where(graph.ndata['attr'][:, 0] == 1)[0]
            return grids, grid_indices
        recidx = np.where(graph.ndata['attr'][:, 0] == 0)[0]
        grididx = np.where(graph.ndata['attr'][:, 0] == 1)[0]
        all_xyz = graph.ndata['x'].squeeze(1).numpy()
        kd_g = scipy.spatial.cKDTree(all_xyz[grididx])
        kd_r = scipy.spatial.cKDTree(all_xyz[recidx])
        firstshell_idx = np.unique(np.concatenate(kd_g.query_ball_tree(kd_r, 4.5)))
        extended_grididx = np.concatenate([grididx, firstshell_idx]).astype(np.int32)
        extended_grids = all_xyz[extended_grididx]
        return extended_grids, extended_grididx

    def _distance_feature(self, distances: torch.Tensor, binsize: float = 0.5,
                         maxd: float = 5.0) -> torch.Tensor:
        d0 = 0.5 * (maxd - binsize)
        m = 5.0 / d0
        feat = 1.0 / (1.0 + torch.exp(-m * (distances - d0)))
        return feat.repeat(1, 3)

class TrainingDataSet(torch.utils.data.Dataset):
    def __init__(self, targets: List[str], ligands: List = None,
                 config: Config = None): 
        """
        Initialize training dataset with grouped parameters

        Args:
            targets: List of target identifiers
            ligands: List of ligand specifications
            config: Main configuration (from configs.config)
        """
        if config is None:
            raise ValueError("Config object must be provided to TrainingDataSet.")
        self.config = config

        self.targets = targets
        self.ligands = ligands
        self.decoys = self._load_decoys()

        # Initialize components by passing relevant sub-configs
        self.loader = MolecularLoader(
            config_paths=self.config.paths,
            config_processing=self.config.processing,
            config_augmentation=self.config.augmentation # Passed to loader for randomize_grid
        )
        self.graph_builder = GraphBuilder(
            config_graph=self.config.graph,
            config_augmentation=self.config.augmentation,
            config_processing=self.config.processing # Passed to graph_builder for drop_H in ligand graph
        )

        self.crossactives = self._load_crossactives() if self.config.cross_validation.load_cross else {}
        self.affinities = self._load_affinities() if self.config.paths.affinityf else {}

        logger.info(f"Training dataset initialized with {len(targets)} targets")
        logger.info(f"Graph config: {self.config.graph}")
        logger.info(f"Processing config: {self.config.processing}")
        logger.info(f"Augmentation config: {self.config.augmentation}")

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> Optional[Tuple]:
        try:
            return self._get_item_safe(index)
        except Exception as e:
            logger.error(f"Error processing item {index}: {e}")
            # traceback.print_exc() # Uncomment for more detailed debug
            return self._get_null_result(self.targets[index])

    def _get_item_safe(self, index: int) -> Optional[Tuple]:
        target = self.targets[index]
        source = target.split('/')[0]
        pname = target.split('/')[1]

        receptor_file_paths = self._setup_file_paths(pname, source)
        if not self._validate_files(receptor_file_paths):
            return self._get_null_result(pname)

        keyatomf = self.loader.find_keyatomf(pname, source)
        if not keyatomf:
            return self._get_null_result(pname)
        keyatoms_dict = self.loader.load_keyatoms(keyatomf)

        grid_data = self._load_grid_data(receptor_file_paths['gridinfo'])
        if grid_data is None:
            return self._get_null_result(pname)
        grids, cats, mask = grid_data

        # Use augmentation config directly
        if self.config.augmentation.randomize_grid > 1e-3:
            grids = self._apply_grid_randomization(grids)

        ligand_result = self._process_ligands(target, self.ligands[index], keyatoms_dict)
        if ligand_result is None:
            return self._get_null_result(pname)
        ligand_graphs, native_graph, key_xyz, key_indices, binding_labels, ligand_info = ligand_result

        origin = torch.tensor(np.mean(grids, axis=0)).float()

        if native_graph is not None:
            native_graph.ndata['x'] = native_graph.ndata['x'] - origin
            key_xyz = key_xyz - origin

        grids = grids - origin.squeeze().numpy()
        gridchain = None # Still using None for now
        receptor_graph, processed_grids, grid_indices = self.graph_builder.build_receptor_graph(
            receptor_file_paths['propnpz'], grids, origin, gridchain
        )

        if receptor_graph is None:
            return self._get_null_result(pname)

        if (receptor_graph.number_of_edges() > self.config.graph.maxedge or
            receptor_graph.number_of_nodes() > self.config.graph.maxnode):
            logger.error(f"Graph too large for {target}: {receptor_graph.number_of_edges()} edges, {receptor_graph.number_of_nodes()} nodes")
            return self._get_null_result(pname)

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
        required_files = ['gridinfo', 'propnpz']
        for file_key in required_files:
            if not os.path.exists(file_paths[file_key]):
                logger.error(f"Required file missing: {file_paths[file_key]}")
                return False
        return True

    def _load_grid_data(self, gridinfo_path: str) -> Optional[Tuple]:
        try:
            sample = np.load(gridinfo_path, allow_pickle=True)
            grids = sample['xyz']
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
        rand_xyz = 2.0 * self.config.augmentation.randomize_grid * (0.5 - np.random.rand(len(grids), 3))
        return grids + rand_xyz

    def _process_ligands(self, target: str, ligands: List,
                        keyatoms_dict: Dict) -> Optional[Tuple]:
        try:
            ligands_parsed, mol2_type, data_type = self._parse_ligands(target, ligands)
            if ligands_parsed is None:
                return None

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

            actives = [ligands_parsed[0]]
            binding_labels = [1 if tag in actives else 0 for tag in tags_read]

            key_xyz = torch.zeros((4, 3))
            if actives and actives[0] in tags_read:
                active_idx = tags_read.index(actives[0])
                native_graph = native_graphs[active_idx] if native_graphs else None
                if not isinstance(native_graph, list): # Check to ensure it's a single graph not a list of graphs
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
        source = target.split('/')[0]
        pname = target.split('/')[1]

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
                active = [active_and_decoys[0]]
                decoys = active_and_decoys[1:]
            else:
                return None, None, None
        else:
            raise ValueError(f"Unknown mol2_type: {mol2_type}")

        if len(decoys) > self.config.processing.max_subset - 1:
            decoys = list(np.random.choice(decoys, self.config.processing.max_subset - len(active), replace=False))

        final_ligands = active + decoys
        return final_ligands, mol2_type, data_type

    def _load_batch_ligands(self, target: str, ligands: List[str],
                          keyatoms_dict: Dict) -> Optional[Tuple]:
        source = target.split('/')[0]
        pname = target.split('/')[1]
        active_ligand = ligands[0]
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
                    mol_tuple = (elem, q, bond, border, coord, nneigh, atype)
                    graph = self.graph_builder.build_ligand_graph(mol_tuple, name=tag)

                    if graph is None:
                        continue

                    com = torch.mean(graph.ndata['x'], axis=0).float()
                    graph.ndata['x'] = (graph.ndata['x'] - com).float()

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

            native_graphs = ligand_graphs

            return ligand_graphs, native_graphs, key_indices_list, atoms_list, tags_processed

        except Exception as e:
            logger.error(f"Error loading batch ligands from {mol2_file}: {e}")
            return None

    def _load_single_ligands(self, target: str, ligands: List[str], keyatoms_dict: Dict) -> Optional[Tuple]:
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
                    if mol2_path.endswith('.mol2'):
                        mol2_file = mol2_path
                        conf_mol2 = mol2_path[:-5] + '.conformers.mol2'
                    else:
                        mol2_file = os.path.join(mol2_path, f'{ligand}.ligand.mol2')
                        conf_mol2 = os.path.join(mol2_path, f'{ligand}.conformers.mol2')

                    clean_ligand = lig_pname

                    mol_data = self.loader.read_mol2_single(mol2_file)
                    if mol_data is None:
                        logger.warning(f"Failed to read {mol2_file}")
                        continue

                    ligand_graph = self.graph_builder.build_ligand_graph(mol_data, name=clean_ligand)
                    if ligand_graph is None:
                        continue

                    com = torch.mean(ligand_graph.ndata['x'], axis=0).float()
                    ligand_graph.ndata['x'] = (ligand_graph.ndata['x'] - com).float()

                    native_graph = copy.deepcopy(ligand_graph)
                    # Use augmentation config directly
                    if (os.path.exists(conf_mol2) and self.config.augmentation.pert and
                        not conf_mol2.endswith(mol2_file)):
                        try:
                            conf_xyz, _ = myutils.read_mol2s_xyzonly(conf_mol2)
                            if conf_xyz:
                                conf_idx = min(np.random.randint(len(conf_xyz)), len(conf_xyz) - 1)
                                native_graph.ndata['x'] = torch.tensor(conf_xyz[conf_idx]).float()[:, None, :]
                        except:
                            pass

                    atoms = mol_data[6]
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
        if target not in keyatoms_dict:
            return []
        key_indices = [atoms.index(atom) for atom in keyatoms_dict[target] if atom in atoms]
        if len(key_indices) > 10:
            key_indices = list(np.random.choice(key_indices, 10, replace=False))
        return key_indices

    def _build_info_dict(self, pname: str, target: str, origin: torch.Tensor,
                        grids: np.ndarray, grid_indices: np.ndarray, gridinfo: str,
                        ligand_info: Dict) -> Dict:
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
        info.update(ligand_info)
        return info

    def _load_decoys(self) -> Dict:
        decoys = {}
        file_path = os.path.join(self.config.paths.datapath, self.config.paths.decoyf)
        if os.path.exists(file_path):
            logger.info(f"Loading decoy NPZ: {file_path}")
            decoys = np.load(file_path, allow_pickle=True)['decoys'].item()
        return decoys

    def _load_crossactives(self) -> Dict:
        mypath = os.path.dirname(os.path.abspath(__file__))
        cross_file = os.path.join(mypath, '../../data/crossreceptor.filtered.npz')
        if os.path.exists(cross_file):
            return np.load(cross_file, allow_pickle=True)['crossrec'].item()
        return {}

    def _load_affinities(self) -> Dict:
        if not self.config.paths.affinityf:
            return {}
        try:
            return parse_affinity(self.config.paths.affinityf)
        except Exception as e:
            logger.error(f"Error loading affinity file {self.config.paths.affinityf}: {e}")
            return {}

    def _get_null_result(self, pname: str) -> Tuple:
        info = {'pname': pname}
        return (None, None, None, None, None, None, None, info)

def collate(samples):
    valid_samples = [s for s in samples if s is not None and s[0] is not None]
    if not valid_samples:
        return None
    receptor_graphs = [s[0] for s in valid_samples]
    ligand_graphs = [s[1] for s in valid_samples]
    cats = [s[2] for s in valid_samples]
    masks = [s[3] for s in valid_samples]
    key_xyz = [s[4] for s in valid_samples]
    key_indices = [s[5] for s in valid_samples]
    binding_labels = [s[6] for s in valid_samples]
    info_dicts = [s[-1] for s in valid_samples]
    batched_receptors = dgl.batch(receptor_graphs)
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
    combined_info = {}
    for key in info_dicts[0]:
        combined_info[key] = [info_dict[key] for info_dict in info_dicts]
    combined_info['grid'] = torch.tensor(np.array(combined_info['grid']))
    batched_grid_indices = []
    node_offset = 0
    for num_nodes, grid_idx in zip(batched_receptors.batch_num_nodes(), combined_info['grididx']):
        batched_grid_indices.append(torch.tensor(grid_idx, dtype=int) + node_offset)
        node_offset += num_nodes
    combined_info['grididx'] = torch.cat(batched_grid_indices, dim=0)
    if ligand_graphs[0]:
        batched_ligands = dgl.batch(ligand_graphs[0])
        global_data = torch.stack([g.gdata for g in ligand_graphs[0]])
        setattr(batched_ligands, "gdata", global_data)
        try:
            batched_key_xyz = key_xyz[0].squeeze()[None, :, :]
            batched_key_indices = [torch.eye(n)[idx] for n, idx in zip(batched_ligands.batch_num_nodes(), key_indices[0])]
            batched_binding_labels = torch.tensor(binding_labels[0], dtype=float)
            combined_info['nK'] = torch.tensor(combined_info['nK'][0])
        except Exception as e:
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

def parse_affinity(affinity_file: str) -> Dict[str, float]:
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
    try:
        with open(active_csv, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 3 and parts[1] == ligand_id:
                    aff_nM = float(parts[2])
                    if aff_nM <= 0:
                        return None
                    aff_M = aff_nM / 1e9
                    return -np.log10(aff_M)
    except Exception as e:
        logger.error(f"Error finding affinity for {ligand_id}: {e}")
    return None

# Backward compatibility alias
DataSet = TrainingDataSet