import os
import numpy as np

from rdkit import Chem
from rdkit.Chem.BRICS import BRICSDecompose, BreakBRICSBonds
from rdkit import RDLogger
from openbabel import pybel 

import multiprocessing as mp
import tempfile

from src.io.ligand_utils import xyz_from_mol2, xyz_from_pdb

RDLogger.DisableLog('rdApp.*')

#######################
# function definition #
#######################

def retrieve_frag_name(input):
    m = Chem.MolFromPDBFile(input)

    if m is None:
        return None
    else:
        orgnames = []
        frag_dic = {}
        for i,atm in enumerate(m.GetAtoms()):
            ri = atm.GetPDBResidueInfo()
            orgnames.append(ri.GetName())
            atm.SetAtomMapNum(i)

        natm = len(m.GetAtoms())
        res = list(BRICSDecompose(m))

        m2 = BreakBRICSBonds(m)
        frags = Chem.GetMolFrags(m2,asMols=True)
        a = 0
        for fragno,f in enumerate(frags):
            frag_atm_list = []
            for atm in f.GetAtoms():
                if atm.GetSymbol() == '*':
                    continue
                i = atm.GetAtomMapNum()
                frag_atm_list.append(orgnames[i].strip())
            frag_dic[fragno] = frag_atm_list
        return frag_dic #{[a1,a2,a3],[]...}


def select_atm_from_frag(xyz,frag_atm_list):
    # xyz are origin-translated
    xyz_f = np.array([xyz[atm] for atm in frag_atm_list])
    com_f = np.mean(xyz_f,axis=0)
    xyz_f -= com_f
    
    d2 = [np.dot(x,x) for x in xyz_f]
    imin = np.argsort(d2)[0]
    selected_atm = frag_atm_list[imin]

    return selected_atm

def frag_from_pdb(pdb, mol2xyz, atms, trg):
    BRICSfragments = retrieve_frag_name(pdb)

    if BRICSfragments == None:
        return

    key_atm_list = []
    for fragno in BRICSfragments:
        fragatms = BRICSfragments[fragno]
        atmxyz = {atm:x for atm,x in zip(atms, mol2xyz)}
        selected_atm = select_atm_from_frag( atmxyz, fragatms)
        key_atm_list.append(selected_atm)

    if len(key_atm_list) < 4:
        npick = 4-len(key_atm_list)
        toadd = list(np.random.choice([a for a in atms if (a not in key_atm_list and a[0] != 'H')],npick))
        key_atm_list += toadd

    return key_atm_list

def split_pdb(pdb, workpath):
    ligpdbs = []
    
    for l in open(pdb):
        if l.startswith('COMPND'):
            tag = l[:-1].split()[-1].replace('.pdb','')
            ligpdb = workpath+'/%s.pdb'%tag
            ligpdbs.append([tag,ligpdb])
            out = open(workpath+'/%s.pdb'%tag,'w')
        if l.startswith('ENDMDL'):
            out.close()
        if l.startswith('ATOM') or l.startswith('CONECT'):
            out.write(l)
    return ligpdbs

##############################
# save key atoms using BRICS #
##############################

def local_runner(args):
    pdb,xyz,atms,trg = args
    try:
        keyatoms = {trg:frag_from_pdb(pdb, xyz, atms, trg)}
    except Exception as e:
        return None
    return keyatoms
    
def launch_batched_ligand(ligand_f, N=10, collated_npz='keyatom.def.npz'):
    if ligand_f.endswith('.mol2'):
        mol2xyz,atms = xyz_from_mol2(ligand_f)
        mol2s = pybel.readfile("mol2", ligand_f)
        ligand_f = ligand_f.replace('.mol2','.pdb')
        ligand_batch_pdb = pybel.Outputfile("pdb", ligand_f, overwrite=True)
        for mol2 in mol2s:
            ligand_batch_pdb.write(mol2)
        ligand_batch_pdb.close()
    elif ligand_f.endswith('.pdb'):
        mol2xyz,atms = xyz_from_pdb(ligand_f) 

    workpath = tempfile.mkdtemp()
    ligpdbs = split_pdb(ligand_f, workpath)

    args = []
    for trg,pdb in ligpdbs:
        args.append((pdb,mol2xyz[trg],atms[trg],trg))
        
    a = mp.Pool(processes=N)
    ans = a.map(local_runner,args)
    keyatms = {}
    for an in ans:
        if an is None: 
            continue
        for tag in an:
            keyatms[tag] = an[tag]
        
    os.system('rm -rf %s'%workpath)
    np.savez(collated_npz,keyatms=keyatms) #keyatm may contain multiple entries for VS
 
if __name__ == "__main__":
    lig_file = "/home/j2ho/DB/for_msk/chembl/Q14432/batch_mol2s_d3/CHEMBL2041604_b.mol2" #sys.argv[1]
    launch_batched_ligand(lig_file, N=1, collated_npz='keyatom.def.npz')

