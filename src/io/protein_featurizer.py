import numpy as np
import copy
import scipy
from pathlib import Path
import logging

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

try:
    import protein_utils as myutils
except ImportError:
    import src.io.protein_utils as myutils

logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(levelname)s] %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

class GridOption:
    def __init__(self,padding,gridsize,option,clash):
        self.padding = padding
        self.gridsize = gridsize
        self.option = option
        self.clash = clash
        self.shellsize=7.0 # throw if no contact within this distance

def read_mol2(mol2f):
    
    from data.generic_potential.Molecule import MoleculeClass
    from data.generic_potential.BasicClasses import OptionClass

    option = OptionClass(['','-s',mol2f])
    molecule = MoleculeClass(mol2f,option)

    option = OptionClass(['','-s',mol2f])
    molecule = MoleculeClass(mol2f,option)

    xyz_lig = np.array(molecule.xyz)
    atypes_lig = [atm.aclass for atm in molecule.atms]
    bases = [atm.root for atm in molecule.atms]

    vbase_lig = [np.array(molecule.xyz[i])-np.array(molecule.xyz[b]) for i,b in enumerate(bases)]
    vbase_lig = np.array([v/(np.linalg.norm(v)+0.001) for v in vbase_lig])
    
    anames_lig = np.array([atm.name for atm in molecule.atms])
    
    return xyz_lig, atypes_lig, molecule.atms_aro, vbase_lig, anames_lig

def sasa_from_xyz(xyz, elems, probe_radius=1.4, n_samples=50):
    atomic_radii = {"C":  2.0,"N": 1.5,"O": 1.4,"S": 1.85,"H": 0.0, #ignore hydrogen for consistency
                    "F": 1.47,"Cl":1.75,"Br":1.85,"I": 2.0,'P': 1.8}
    
    areas = []
    normareas = []
    centers = xyz
    radii = np.array([atomic_radii[e] for e in elems])

    inc = np.pi * (3 - np.sqrt(5)) # increment
    off = 2.0/n_samples

    pts0 = []
    for k in range(n_samples):
        phi = k * inc
        y = k * off - 1 + (off / 2)
        r = np.sqrt(1 - y*y)
        pts0.append([np.cos(phi) * r, y, np.sin(phi) * r])
    pts0 = np.array(pts0)

    kd = scipy.spatial.cKDTree(xyz)
    neighs = kd.query_ball_tree(kd, 8.0)

    occls = []
    for i,(neigh, center, radius) in enumerate(zip(neighs, centers, radii)):
        neigh.remove(i)
        n_neigh = len(neigh)
        d2cen = np.sum((center[None,:].repeat(n_neigh,axis=0) - xyz[neigh]) ** 2, axis=1)
        occls.append(d2cen)
        
        pts = pts0*(radius+probe_radius) + center
        n_neigh = len(neigh)
        
        x_neigh = xyz[neigh][None,:,:].repeat(n_samples,axis=0)
        pts = pts.repeat(n_neigh, 0).reshape(n_samples, n_neigh, 3)
        
        d2 = np.sum((pts - x_neigh) ** 2, axis=2) # Here. time-consuming line
        r2 = (radii[neigh] + probe_radius) ** 2
        r2 = np.stack([r2] * n_samples)

        # If probe overlaps with just one atom around it, it becomes an insider
        n_outsiders = np.sum(np.all(d2 >= (r2 * 0.99), axis=1))  # the 0.99 factor to account for numerical errors in the calculation of d2
        # The surface area of   the sphere that is not occluded
        area = 4 * np.pi * ((radius + probe_radius) ** 2) * n_outsiders / n_samples
        areas.append(area)

        norm = 4 * np.pi * (radius + probe_radius)
        normareas.append(min(1.0,area/norm))

    occls = np.array([np.sum(np.exp(-occl/6.0),axis=-1) for occl in occls])
    occls = (occls-6.0)/3.0 #rerange 3.0~9.0 -> -1.0~1.0
    return areas, np.array(normareas), occls

def featurize_target_properties(pdb,npz,extrapath="",verbose=False):
    # get receptor info
    qs_aa, atypes_aa, atms_aa, bnds_aa, repsatm_aa = myutils.get_AAtype_properties(extrapath=extrapath,
                                                                                   extrainfo={})
    resnames,reschains,xyz,atms = myutils.read_pdb(pdb,read_ligand=False)
    
    # read in only heavy + hpol atms as lists
    q_rec = []
    atypes_rec = []
    xyz_rec = []
    atmres_rec = []
    aas_rec = []
    bnds_rec = []
    repsatm_idx = []
    residue_idx = []
    atmnames = []

    skipres = []
    reschains_read = []
    reschains_idx = []

    for i,resname in enumerate(resnames):
        reschain = reschains[i]

        if resname in myutils.ALL_AAS:
            iaa = myutils.ALL_AAS.index(resname)
            qs, atypes, atms, bnds_, repsatm = (qs_aa[iaa], atypes_aa[iaa], atms_aa[iaa], bnds_aa[iaa], repsatm_aa[iaa])
        else:
            if verbose: 
                logger.warning("Unknown residue: %s, skipping..." % resname)
            skipres.append(i)
            continue

        natm = len(xyz_rec)
        atms_r = []
                
        # unify metal index to Calcium for simplification
        if iaa >= myutils.ALL_AAS.index("CA"):
            iaa = myutils.ALL_AAS.index("CA")
            atypes = atypes_aa[iaa]
            
        for iatm,atm in enumerate(atms):
            is_repsatm = (iatm == repsatm)
            
            if atm not in xyz[reschain]:
                if is_repsatm: return False
                continue
                
            atms_r.append(atm)
            q_rec.append(qs[atm])
            atypes_rec.append(atypes[iatm])
            aas_rec.append(iaa)
            xyz_rec.append(xyz[reschain][atm])
            atmres_rec.append((reschain,atm))
            residue_idx.append(i)
            if is_repsatm: repsatm_idx.append(natm+iatm)

        bnds = [[atms_r.index(atm1),atms_r.index(atm2)] for atm1,atm2 in bnds_ if atm1 in atms_r and atm2 in atms_r]

        # make sure all bonds are right
        for (i1,i2) in copy.copy(bnds):
            dv = np.array(xyz_rec[i1+natm]) - np.array(xyz_rec[i2+natm])
            d = np.sqrt(np.dot(dv,dv))
            if d > 2.0:
                if verbose:
                    logger.warning(f"Warning, abnormal bond distance: {pdb} {resname} {reschain} {i1} {i2} {atms_r[i1]} {atms_r[i2]} {d:.2f}")
                bnds.remove([i1,i2])
                
        bnds = np.array(bnds,dtype=int)
        atmnames += atms_r
        reschains_idx += [reschain for _ in atms_r]
        reschains_read.append(reschain)

        if i == 0:
            bnds_rec = bnds
        elif bnds_ != []:
            bnds += natm
            bnds_rec = np.concatenate([bnds_rec,bnds])
            
    xyz_rec = np.array(xyz_rec)

    if len(atmnames) != len(xyz_rec):
        sys.exit('inconsistent anames <=> xyz')

    elems_rec = [at[0] for at in atypes_rec]
    sasa, normsasa, occl = sasa_from_xyz(xyz_rec, elems_rec)
    
    np.savez(npz,
             # per-atm
             aas_rec=aas_rec,
             xyz_rec=xyz_rec, #just native info
             atypes_rec=atypes_rec, #string
             charge_rec=q_rec,
             bnds_rec=bnds_rec,
             sasa_rec=normsasa, #apo
             occl=occl,
             residue_idx=residue_idx,
             atmres_rec=atmres_rec,
             atmnames=atmnames, #[[],[],[],...]
                 
             # per-res (only for receptor)
             repsatm_idx=repsatm_idx,
             reschains=reschains,

        )

    return xyz_rec, aas_rec, atmres_rec, atypes_rec, q_rec, bnds_rec, sasa, residue_idx, repsatm_idx, reschains, reschains_idx, atmnames

def grid_from_xyz(xyzs_rec,xyzs_lig,
                  opt,
                  gridout=None,
                  verbose=False):

    bmin = np.min(xyzs_lig[:,:]-opt.padding,axis=0)
    bmax = np.max(xyzs_lig[:,:]+opt.padding,axis=0)

    imin = [int(bmin[k]/opt.gridsize)-1 for k in range(3)]
    imax = [int(bmax[k]/opt.gridsize)+1 for k in range(3)]

    grids = []
    if verbose:
        logger.info("Detected %d grid points..."%((imax[0]-imin[0])*(imax[1]-imin[1])*(imax[2]-imin[2])))
    for ix in range(imin[0],imax[0]+1):
        for iy in range(imin[1],imax[1]+1):
            for iz in range(imin[2],imax[2]+1):
                grid = np.array([ix*opt.gridsize,iy*opt.gridsize,iz*opt.gridsize])
                grids.append(grid)

    grids = np.array(grids)
    nfull = len(grids)

    # Remove clashing or far-off grids
    kd      = scipy.spatial.cKDTree(grids)
    kd_ca   = scipy.spatial.cKDTree(xyzs_rec)
    kd_lig  = scipy.spatial.cKDTree(xyzs_lig)

    # take ligand-neighs
    excl = np.concatenate(kd_ca.query_ball_tree(kd, opt.clash)) #clashing
    incl = np.unique(np.concatenate(kd_ca.query_ball_tree(kd, opt.shellsize))) #grid-rec shell
    ilig = np.unique(np.concatenate(kd_lig.query_ball_tree(kd, opt.padding))) # ligand environ

    interface = np.unique(np.array([i for i in incl if (i not in excl and i in ilig)],dtype=np.int16))
    grids = grids[interface]
    n1 = len(grids)

    # filter small segments by default
    D = scipy.spatial.distance_matrix(grids,grids)
    graph = csr_matrix((D<(opt.gridsize+0.1)).astype(int))
    n, labels = connected_components(csgraph=graph, directed=False, return_labels=True)

    ncl = [sum(labels==k) for k in range(n)]
    biggest = np.unique(np.where(labels==np.argmax(ncl)))
    
    grids = grids[biggest]
    if verbose:
        logger.info("Search through %d grid points, of %d contact grids %d clash -> %d, remove outlier -> %d"%(nfull,len(incl),len(excl),n1,len(grids)))
        # logger doesn't get printed in stdout?
    if gridout is not None:
        for i,grid in enumerate(grids):
            gridout.write("HETATM %4d  CA  CA  X   1    %8.3f%8.3f%8.3f\n"%(i,grid[0],grid[1],grid[2]))
    return grids

def get_motifs_from_complex(xyz_rec, atypes_rec, xyzs_lig, atypes_lig, aroatms_lig, vbase_lig,
                            mode='gen', debug=False, outprefix=''):

    # ligand -- gentype
    if mode == 'gen':
        donorclass_gen = [21,22,23,25,27,28,31,32,34,43] 
        acceptorclass_gen = [22,26,30,33,34,36,37,38,39,41,42,43,47]
        aliphaticclass_gen = [3,4] #3: CH2, 4: CH3; -> make [4] to be more strict (only CH3)

        D_lig = [i for i,at in enumerate(atypes_lig) if at in donorclass_gen]
        A_lig = [i for i,at in enumerate(atypes_lig) if at in acceptorclass_gen]
        H_lig = [i for i,at in enumerate(atypes_lig) if at in aliphaticclass_gen]
        R_lig = [i for i,at in enumerate(atypes_lig) if i in aroatms_lig]
        
    else: # processed types
        D_lig = [i for i,at in enumerate(atypes_lig) if at in [1,3]]
        A_lig = [i for i,at in enumerate(atypes_lig) if at in [1,2]]
        H_lig = [i for i,at in enumerate(atypes_lig) if at==4]
        R_lig = [i for i,at in enumerate(atypes_lig) if at==5]

    donorclass_aa = ['Ohx','Nad','Nim','Ngu1','Ngu2','Nam','Ca2p','Mg2p','Mn','Fe2p','Zn2p','Co2p','Cu2p','Ni','Cd'] #metals are considerec Ca2p anyways...
    acceptorclass_aa = ['Oad','Oat','Ohx','Nin']
    HRclass_aa = ['CS3','CS2','CR','CRp']

    D_rec = [i for i,at in enumerate(atypes_rec) if at in donorclass_aa]
    A_rec = [i for i,at in enumerate(atypes_rec) if at in acceptorclass_aa]
    HR_rec = [i for i,at in enumerate(atypes_rec) if at in HRclass_aa]

    kd_D   = scipy.spatial.cKDTree(xyz_rec[D_rec])
    kd_A   = scipy.spatial.cKDTree(xyz_rec[A_rec])
    kd_HR   = scipy.spatial.cKDTree(xyz_rec[HR_rec])
    
    kd_lig  = scipy.spatial.cKDTree(xyzs_lig)

    # not super fast but okay
    dv2D = np.array([[y-x for x in xyzs_lig] for y in xyz_rec[D_rec]])
    dv2A = np.array([[y-x for x in xyzs_lig] for y in xyz_rec[A_rec]])

    d2D = np.sqrt(np.einsum('ijk,ijk->ij',dv2D,dv2D))
    d2A = np.sqrt(np.einsum('ijk,ijk->ij',dv2A,dv2A))

    o2D = np.einsum('jk,ijk->ij', vbase_lig, dv2D)/d2D
    o2A = np.einsum('jk,ijk->ij', vbase_lig, dv2A)/d2A

    iA = np.where(((d2D<3.6)*(o2D>0.2588))>0.99)[1] # allow up to 105'
    iD = np.unique(np.where(d2A<3.6)[1])

    iHR = np.unique(np.concatenate(kd_HR.query_ball_tree(kd_lig,5.1)).astype(int))

    motifs = []
    for i,xyz in enumerate(xyzs_lig):
        mtype = 0
        # let mutually exclusive
        if (i in iA or i in iD) and (i in A_lig and i in D_lig): mtype = 1 #Both

        #new -- don't mind if lig-acceptor is close to rec-acceptor & vice versa for donor
        elif (i in iA) and (i in A_lig): mtype = 2 # Acc
        elif (i in iD) and (i in D_lig): mtype = 3 # Don
        elif i in H_lig and i in iHR: mtype = 4 # Ali
        elif i in R_lig and i in iHR: mtype = 5 # Aro

        if mtype > 0:
            if debug:
                print(i, atypes_lig[i], xyz, mtype)
            motifs.append((xyz,mtype,1.0))

    if debug:
        out = open('%s.motif.xyz'%outprefix,'w')
        aname = ['X','B','O','N','C','R']
        for x,m,_ in motifs:
            out.write('%-4s %8.3f %8.3f %8.3f\n'%(aname[m],x[0],x[1],x[2]))
        out.close()

    return motifs

def define_motifs_from_multiple_complex(pdbs, ligmol2s, outprefix, weights='auto', debug=False):
    if weights == 'auto':
        weights = [0.5 for _ in pdbs]
        weights[0] = 1.0
        
    motifs_all = []
    motifs_self = []

    for i,(w,pdb,ligmol2) in enumerate(zip(weights,pdbs,ligmol2s)):
        recnpz = "%s.prop.npz"%(outprefix)
        recnpz_path = Path(recnpz)
        if not recnpz_path.is_file():
            continue
        recdata = np.load(recnpz,allow_pickle=True)
        xyzs_rec = recdata['xyz_rec']
        atypes_rec = recdata['atypes_rec']
        xyzs_lig, atypes_lig, aroatms_lig, vbase_lig, anames_lig = read_mol2(ligmol2)
    
        motifs = get_motifs_from_complex(xyzs_rec, atypes_rec,
                                         xyzs_lig, atypes_lig, aroatms_lig, vbase_lig,
                                         anames_lig,
                                         mode='gen',
                                         debug=debug,outprefix=outprefix)

        motifs = [(m[0],m[1],w) for m in motifs]
        motifs_all += motifs
        if i == 0:
            motifs_self += motifs

    return motifs_all, motifs_self

def motif2label(motifs_all, grids, outprefix='', debug=False, sig=1.0):
    # distance b/w grid & true-labeled-motifs
    dv2xyz = np.array([[g-x for g in grids] for x,c,_ in motifs_all]) # grids x numTrue
    d2xyz = np.sum(dv2xyz*dv2xyz,axis=2)

    overlap = np.exp(-d2xyz/sig/sig)

    label = np.zeros((len(grids),6))

    for o,(_,cat,w) in zip(overlap,motifs_all): # motif index
        for j,p in enumerate(o): # grid index
            if p > 0.01:
                label[j,cat] = max(label[j,cat],np.sqrt(p))

    if debug and outprefix != '':
        out = open(outprefix+'.label.pdb','w')
        nlabeled = 0
        for i,l in enumerate(label):
            grid = grids[i]
            if max(l) > 0.01:
                imotif = np.where(l>0.01)[0]
                for j in imotif:
                    B = np.sqrt(l[j])
                    nlabeled += 1
                    mname = ['H','CB','CA','CD','CH','CR'][j]
                    out.write("HETATM %4d  %2s  %2s  X%4d    %8.3f%8.3f%8.3f  1.00  %5.2f\n"%(i,mname,mname,i,grid[0],grid[1],grid[2],B))
            else:
                out.write("HETATM %4d  H   H   X%4d    %8.3f%8.3f%8.3f  1.00  %5.2f\n"%(i,i,grid[0],grid[1],grid[2],0.0))
        out.close()
        
    return label

def define_label_from_ligand(grids, xyzs_rec, atypes_rec, ligmol2,
                             debug=False, outprefix=''):
    xyzs_lig, atypes_lig, aroatms_lig, vbase_lig, anames_lig = read_mol2(ligmol2)
    
    motifs = get_motifs_from_complex(xyzs_rec, atypes_rec,
                                     xyzs_lig, atypes_lig, aroatms_lig, vbase_lig,
                                     anames_lig,
                                     mode='gen',
                                     debug=debug,outprefix=outprefix)
    
    # distance b/w grid & true-labeled-motifs
    dv2xyz = np.array([[g-x for g in grids] for x,c,_ in motifs]) # grids x numTrue
    d2xyz = np.sum(dv2xyz*dv2xyz,axis=2)

    sig = 1.0
    overlap = np.exp(-d2xyz/sig/sig)

    label = np.zeros((len(grids),6))

    for o,(_,cat,_) in zip(overlap,motifs): # motif index
        for j,p in enumerate(o): # grid index
            if p > 0.01:
                label[j,cat] = max(label[j,cat],np.sqrt(p))

    if outprefix != '':
        out = open(outprefix+'.label.pdb','w')
        nlabeled = 0
        for i,l in enumerate(label):
            grid = grids[i]
            if max(l) > 0.01:
                imotif = np.where(l>0.01)[0]
                for j in imotif:
                    B = np.sqrt(l[j])
                    nlabeled += 1
                    mname = ['H','CB','CA','CD','CH','CR'][j]
                    out.write("HETATM %4d  %2s  %2s  X%4d    %8.3f%8.3f%8.3f  1.00  %5.2f\n"%(i,mname,mname,i,grid[0],grid[1],grid[2],B))
            else:
                out.write("HETATM %4d  H   H   X%4d    %8.3f%8.3f%8.3f  1.00  %5.2f\n"%(i,i,grid[0],grid[1],grid[2],0.0))
        out.close()

    return label

def main(pdb,outprefix,
         recpdb=None,
         ligmol2='',
         gridsize=1.5,
         padding=10.0,
         clash=1.1,
         gridoption='com',
         maskres=[],
         com=[],
         verbose=True):
    # read relevant motif
    aas, reschains, xyz, atms = myutils.read_pdb(pdb,read_ligand=True)
    
    gridopt = GridOption(padding,gridsize,gridoption,clash)
    
    xyz = [np.array(list(xyz[rc].values()),dtype=np.float32) for rc in reschains if rc not in maskres]
    xyz = np.concatenate(xyz)

    # com should be passed through input argument
    assert(len(com) == 3)
    com = com[None,:]
    
    grids = grid_from_xyz(xyz,com,gridopt,verbose=verbose)
       
    recnpz = "%s.prop.npz"%(outprefix)
    if recpdb is None:
        recpdb = pdb
    if verbose:
        logger.info("Featurize receptor info from %s...\n"%recpdb)
    args = featurize_target_properties(recpdb,recnpz,verbose=verbose)
    xyz_rec = args[0]
    atypes_rec = args[3]
    
    xyzs = []
    tags = []
    for i,grid in enumerate(grids):
        xyzs.append(grid)
        tags.append("grid%04d"%i)

    label = []
    if ligmol2 != '':
        label = define_label_from_ligand(grids, xyz_rec, atypes_rec, ligmol2)
        
    gridnpz = "%s.grid.npz"%(outprefix)
    if len(label) == 0:
        np.savez(gridnpz, xyz=xyzs, name=tags)
    else:
        np.savez(gridnpz, xyz=xyzs, name=tags, labels=label)


def calculate_ligand_com(ligand_file):
    """Calculate center of mass from ligand file (PDB or MOL2) using ligand_utils functions"""
    try:
        from ligand_utils import xyz_from_mol2, xyz_from_pdb
    except ImportError:
        from src.io.ligand_utils import xyz_from_mol2, xyz_from_pdb
    
    ligand_file = Path(ligand_file)
    
    if ligand_file.suffix.lower() == '.pdb':
        xyz_dict, _ = xyz_from_pdb(str(ligand_file), centered=False)
        
        if not xyz_dict:
            raise ValueError(f"No coordinates found in ligand PDB: {ligand_file}")
        
        coords = list(xyz_dict.values())
        com = np.mean(coords, axis=0)
        
    elif ligand_file.suffix.lower() == '.mol2':
        xyz_dict, _ = xyz_from_mol2(str(ligand_file), centered=False)
        
        if not xyz_dict:
            raise ValueError(f"No coordinates found in ligand MOL2: {ligand_file}")
        
        first_mol = list(xyz_dict.keys())[0]
        coords = xyz_dict[first_mol]
        com = np.mean(coords, axis=0)
        
    else:
        raise ValueError(f"Unsupported ligand file format: {ligand_file.suffix}. Supported: .pdb, .mol2")
    
    return com

mode = 'com'
def runner(config_dict):
    """
    Enhanced runner that accepts a config dictionary with:
    - protein_pdb: path to protein PDB file
    - center: [x,y,z] coordinates (optional)
    - crystal_ligand: path to crystal ligand file (optional)
    - gridsize: grid spacing (default 1.5)
    - padding: padding around center (default 10.0) 
    - clash: clash distance (default 1.1)
    
    Priority: if both center and crystal_ligand provided, center takes priority
    """
    pdb = config_dict['protein_pdb']
    center_coords = config_dict.get('center')
    crystal_ligand = config_dict.get('crystal_ligand')
    gridsize = config_dict.get('gridsize', 1.5)
    padding = config_dict.get('padding', 10.0)
    clash = config_dict.get('clash', 1.1)
    output_dir = config_dict.get('output_dir', '.')
    logger.info(f"Running protein featurizer with receptor: {pdb}")
    outprefix = Path(output_dir) / Path(pdb).stem
    logger.info(f"Processed protein grid files will be: {outprefix}.prop.npz and {outprefix}.grid.npz")
    
    # Determine center of mass - center takes priority over crystal_ligand
    if center_coords is not None:
        # User provided coordinates - highest priority
        if isinstance(center_coords, (list, tuple)) and len(center_coords) == 3:
            com = np.array(center_coords, dtype=float)
            logger.info(f"Using user-provided center: {com}")
        else:
            raise ValueError("center must be a list/tuple of 3 coordinates [x,y,z]")
    elif crystal_ligand is not None:
        # Path to crystal ligand file - second priority
        ligand_file = Path(crystal_ligand)
        if not ligand_file.exists():
            logger.error(f"Crystal ligand file not found: {ligand_file}")
            raise FileNotFoundError(f"Crystal ligand file not found: {ligand_file}")
        
        com = calculate_ligand_com(ligand_file)
        logger.info(f"Calculated center from crystal ligand {ligand_file}: {com}")
    else:
        logger.error("Neither 'center' coordinates nor 'crystal_ligand' file path provided")
        raise ValueError("Either 'center' coordinates [x,y,z] or 'crystal_ligand' file path must be provided")
    
    main(pdb,
         outprefix=str(outprefix),
         gridsize=gridsize,
         com=com,
         padding=padding,
         clash=clash,
         gridoption='com')
    
    return str(outprefix)

if __name__ == "__main__":
    config = {
        'protein_pdb': '/home/j2ho/DB/dud-e/aa2ar/receptor.pdb',
        'crystal_ligand': 'actives_final.mol2'
    }
    runner(config)
    