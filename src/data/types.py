import numpy as np
import glob
import os
import torch

ELEMS = ['Null','H','C','N','O','Cl','F','I','Br','P','S'] #0 index goes to "empty node"

# Tip atom definitions
AA_to_tip = {"ALA":"CB", "CYS":"SG", "ASP":"CG", "ASN":"CG", "GLU":"CD",
             "GLN":"CD", "PHE":"CZ", "HIS":"NE2", "ILE":"CD1", "GLY":"CA",
             "LEU":"CG", "MET":"SD", "ARG":"CZ", "LYS":"NZ", "PRO":"CG",
             "VAL":"CB", "TYR":"OH", "TRP":"CH2", "SER":"OG", "THR":"OG1"}

# Residue number definition
AMINOACID = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU',\
             'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE',\
             'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']
residuemap = dict([(AMINOACID[i], i) for i in range(len(AMINOACID))])
NUCLEICACID = ['ADE','CYT','GUA','THY','URA'] #nucleic acids

METAL = ['CA','ZN','MN','MG','FE','CD','CO','CU']
ALL_AAS = ['UNK'] + AMINOACID + NUCLEICACID + METAL
NMETALS = len(METAL)

N_AATYPE = len(ALL_AAS)

# minimal sc atom representation (Nx8)
aa2short={
    "ALA": (" N  "," CA "," C  "," CB ",  None,  None,  None,  None), 
    "ARG": (" N  "," CA "," C  "," CB "," CG "," CD "," NE "," CZ "), 
    "ASN": (" N  "," CA "," C  "," CB "," CG "," OD1",  None,  None), 
    "ASP": (" N  "," CA "," C  "," CB "," CG "," OD1",  None,  None), 
    "CYS": (" N  "," CA "," C  "," CB "," SG ",  None,  None,  None), 
    "GLN": (" N  "," CA "," C  "," CB "," CG "," CD "," OE1",  None), 
    "GLU": (" N  "," CA "," C  "," CB "," CG "," CD "," OE1",  None), 
    "GLY": (" N  "," CA "," C  ",  None,  None,  None,  None,  None), 
    "HIS": (" N  "," CA "," C  "," CB "," CG "," ND1",  None,  None),
    "ILE": (" N  "," CA "," C  "," CB "," CG1"," CD1",  None,  None), 
    "LEU": (" N  "," CA "," C  "," CB "," CG "," CD1",  None,  None), 
    "LYS": (" N  "," CA "," C  "," CB "," CG "," CD "," CE "," NZ "), 
    "MET": (" N  "," CA "," C  "," CB "," CG "," SD "," CE ",  None), 
    "PHE": (" N  "," CA "," C  "," CB "," CG "," CD1",  None,  None),
    "PRO": (" N  "," CA "," C  "," CB "," CG "," CD ",  None,  None), 
    "SER": (" N  "," CA "," C  "," CB "," OG ",  None,  None,  None),
    "THR": (" N  "," CA "," C  "," CB "," OG1",  None,  None,  None),
    "TRP": (" N  "," CA "," C  "," CB "," CG "," CD1",  None,  None),
    "TYR": (" N  "," CA "," C  "," CB "," CG "," CD1",  None,  None),
    "VAL": (" N  "," CA "," C  "," CB "," CG1",  None,  None,  None),
}

# Atom types:
atypes = {('ALA', 'CA'): 'CAbb', ('ALA', 'CB'): 'CH3', ('ALA', 'C'): 'CObb', ('ALA', 'N'): 'Nbb', ('ALA', 'O'): 'OCbb', ('ARG', 'CA'): 'CAbb', ('ARG', 'CB'): 'CH2', ('ARG', 'C'): 'CObb', ('ARG', 'CD'): 'CH2', ('ARG', 'CG'): 'CH2', ('ARG', 'CZ'): 'aroC', ('ARG', 'NE'): 'Narg', ('ARG', 'NH1'): 'Narg', ('ARG', 'NH2'): 'Narg', ('ARG', 'N'): 'Nbb', ('ARG', 'O'): 'OCbb', ('ASN', 'CA'): 'CAbb', ('ASN', 'CB'): 'CH2', ('ASN', 'C'): 'CObb', ('ASN', 'CG'): 'CNH2', ('ASN', 'ND2'): 'NH2O', ('ASN', 'N'): 'Nbb', ('ASN', 'OD1'): 'ONH2', ('ASN', 'O'): 'OCbb', ('ASP', 'CA'): 'CAbb', ('ASP', 'CB'): 'CH2', ('ASP', 'C'): 'CObb', ('ASP', 'CG'): 'COO', ('ASP', 'N'): 'Nbb', ('ASP', 'OD1'): 'OOC', ('ASP', 'OD2'): 'OOC', ('ASP', 'O'): 'OCbb', ('CYS', 'CA'): 'CAbb', ('CYS', 'CB'): 'CH2', ('CYS', 'C'): 'CObb', ('CYS', 'N'): 'Nbb', ('CYS', 'O'): 'OCbb', ('CYS', 'SG'): 'S', ('GLN', 'CA'): 'CAbb', ('GLN', 'CB'): 'CH2', ('GLN', 'C'): 'CObb', ('GLN', 'CD'): 'CNH2', ('GLN', 'CG'): 'CH2', ('GLN', 'NE2'): 'NH2O', ('GLN', 'N'): 'Nbb', ('GLN', 'OE1'): 'ONH2', ('GLN', 'O'): 'OCbb', ('GLU', 'CA'): 'CAbb', ('GLU', 'CB'): 'CH2', ('GLU', 'C'): 'CObb', ('GLU', 'CD'): 'COO', ('GLU', 'CG'): 'CH2', ('GLU', 'N'): 'Nbb', ('GLU', 'OE1'): 'OOC', ('GLU', 'OE2'): 'OOC', ('GLU', 'O'): 'OCbb', ('GLY', 'CA'): 'CAbb', ('GLY', 'C'): 'CObb', ('GLY', 'N'): 'Nbb', ('GLY', 'O'): 'OCbb', ('HIS', 'CA'): 'CAbb', ('HIS', 'CB'): 'CH2', ('HIS', 'C'): 'CObb', ('HIS', 'CD2'): 'aroC', ('HIS', 'CE1'): 'aroC', ('HIS', 'CG'): 'aroC', ('HIS', 'ND1'): 'Nhis', ('HIS', 'NE2'): 'Ntrp', ('HIS', 'N'): 'Nbb', ('HIS', 'O'): 'OCbb', ('ILE', 'CA'): 'CAbb', ('ILE', 'CB'): 'CH1', ('ILE', 'C'): 'CObb', ('ILE', 'CD1'): 'CH3', ('ILE', 'CG1'): 'CH2', ('ILE', 'CG2'): 'CH3', ('ILE', 'N'): 'Nbb', ('ILE', 'O'): 'OCbb', ('LEU', 'CA'): 'CAbb', ('LEU', 'CB'): 'CH2', ('LEU', 'C'): 'CObb', ('LEU', 'CD1'): 'CH3', ('LEU', 'CD2'): 'CH3', ('LEU', 'CG'): 'CH1', ('LEU', 'N'): 'Nbb', ('LEU', 'O'): 'OCbb', ('LYS', 'CA'): 'CAbb', ('LYS', 'CB'): 'CH2', ('LYS', 'C'): 'CObb', ('LYS', 'CD'): 'CH2', ('LYS', 'CE'): 'CH2', ('LYS', 'CG'): 'CH2', ('LYS', 'N'): 'Nbb', ('LYS', 'NZ'): 'Nlys', ('LYS', 'O'): 'OCbb', ('MET', 'CA'): 'CAbb', ('MET', 'CB'): 'CH2', ('MET', 'C'): 'CObb', ('MET', 'CE'): 'CH3', ('MET', 'CG'): 'CH2', ('MET', 'N'): 'Nbb', ('MET', 'O'): 'OCbb', ('MET', 'SD'): 'S', ('PHE', 'CA'): 'CAbb', ('PHE', 'CB'): 'CH2', ('PHE', 'C'): 'CObb', ('PHE', 'CD1'): 'aroC', ('PHE', 'CD2'): 'aroC', ('PHE', 'CE1'): 'aroC', ('PHE', 'CE2'): 'aroC', ('PHE', 'CG'): 'aroC', ('PHE', 'CZ'): 'aroC', ('PHE', 'N'): 'Nbb', ('PHE', 'O'): 'OCbb', ('PRO', 'CA'): 'CAbb', ('PRO', 'CB'): 'CH2', ('PRO', 'C'): 'CObb', ('PRO', 'CD'): 'CH2', ('PRO', 'CG'): 'CH2', ('PRO', 'N'): 'Npro', ('PRO', 'O'): 'OCbb', ('SER', 'CA'): 'CAbb', ('SER', 'CB'): 'CH2', ('SER', 'C'): 'CObb', ('SER', 'N'): 'Nbb', ('SER', 'OG'): 'OH', ('SER', 'O'): 'OCbb', ('THR', 'CA'): 'CAbb', ('THR', 'CB'): 'CH1', ('THR', 'C'): 'CObb', ('THR', 'CG2'): 'CH3', ('THR', 'N'): 'Nbb', ('THR', 'OG1'): 'OH', ('THR', 'O'): 'OCbb', ('TRP', 'CA'): 'CAbb', ('TRP', 'CB'): 'CH2', ('TRP', 'C'): 'CObb', ('TRP', 'CD1'): 'aroC', ('TRP', 'CD2'): 'aroC', ('TRP', 'CE2'): 'aroC', ('TRP', 'CE3'): 'aroC', ('TRP', 'CG'): 'aroC', ('TRP', 'CH2'): 'aroC', ('TRP', 'CZ2'): 'aroC', ('TRP', 'CZ3'): 'aroC', ('TRP', 'NE1'): 'Ntrp', ('TRP', 'N'): 'Nbb', ('TRP', 'O'): 'OCbb', ('TYR', 'CA'): 'CAbb', ('TYR', 'CB'): 'CH2', ('TYR', 'C'): 'CObb', ('TYR', 'CD1'): 'aroC', ('TYR', 'CD2'): 'aroC', ('TYR', 'CE1'): 'aroC', ('TYR', 'CE2'): 'aroC', ('TYR', 'CG'): 'aroC', ('TYR', 'CZ'): 'aroC', ('TYR', 'N'): 'Nbb', ('TYR', 'OH'): 'OH', ('TYR', 'O'): 'OCbb', ('VAL', 'CA'): 'CAbb', ('VAL', 'CB'): 'CH1', ('VAL', 'C'): 'CObb', ('VAL', 'CG1'): 'CH3', ('VAL', 'CG2'): 'CH3', ('VAL', 'N'): 'Nbb', ('VAL', 'O'): 'OCbb'}

# Atome type to index
atype2num = {'CNH2': 0, 'Npro': 1, 'CH1': 2, 'CH3': 3, 'CObb': 4, 'aroC': 5, 'OOC': 6, 'Nhis': 7, 'Nlys': 8, 'COO': 9, 'NH2O': 10, 'S': 11, 'Narg': 12, 'OCbb': 13, 'Ntrp': 14, 'Nbb': 15, 'CH2': 16, 'CAbb': 17, 'ONH2': 18, 'OH': 19}

gentype2num = {'CS':0, 'CS1':1, 'CS2':2,'CS3':3,
               'CD':4, 'CD1':5, 'CD2':6,'CR':7, 'CT':8,
               'CSp':9,'CDp':10,'CRp':11,'CTp':12,'CST':13,'CSQ':14,
               'HO':15,'HN':16,'HS':17,
               # Nitrogen
               'Nam':18, 'Nam2':19, 'Nad':20, 'Nad3':21, 'Nin':22, 'Nim':23,
               'Ngu1':24, 'Ngu2':25, 'NG3':26, 'NG2':27, 'NG21':28,'NG22':29, 'NG1':30, 
               'Ohx':31, 'Oet':32, 'Oal':33, 'Oad':34, 'Oat':35, 'Ofu':36, 'Ont':37, 'OG2':38, 'OG3':39, 'OG31':40,
               #S/P
               'Sth':41, 'Ssl':42, 'SR':43,  'SG2':44, 'SG3':45, 'SG5':46, 'PG3':47, 'PG5':48, 
               # Halogens
               'Br':49, 'I':50, 'F':51, 'Cl':52, 'BrR':53, 'IR':54, 'FR':55, 'ClR':56,
               # Metals
               'Ca2p':57, 'Mg2p':58, 'Mn':59, 'Fe2p':60, 'Fe3p':60, 'Zn2p':61, 'Co2p':62, 'Cu2p':63, 'Cd':64}

def find_gentype2num(at):
    if at in gentype2num:
        return gentype2num[at]
    else:
        return 0 # is this okay?

# simplified idx
gentype2simple = {'CS':0,'CS1':0,'CS3':0,'CST':0,'CSQ':0,'CSp':0,
                  'CD':1,'CD1':1,'CD2':1,'CDp':1,
                  'CT':2,'CTp':2,
                  'CR':3,'CRp':3,
                  'HN':4,'HO':4,'HS':4,
                  'Nam':5,'Nam2':5,'NG3':5,
                  'Nad':6,'Nad3':6,'Nin':6,'Nim':6,'Ngu1':6,'Ngu2':6,'NG2':6,'NG21':6,'NG22':6,
                  'NG1':7,
                  'Ohx':8,'OG3':8,'Oet':8,'OG31':8,
                  'Oal':9, 'Oad':9, 'Oat':9, 'Ofu':9, 'Ont':9, 'OG2':9,
                  'Sth':10, 'Ssl':10, 'SR':10,  'SG2':10, 'SG3':10, 'SG5':10, 'PG3':11, 'PG5':11, 
                  'F':12, 'Cl':13, 'Br':14, 'I':15, 'FR':12, 'ClR':13, 'BrR':14, 'IR':15, 
                  'Ca2p':16, 'Mg2p':17, 'Mn':18, 'Fe2p':19, 'Fe3p':19, 'Zn2p':20, 'Co2p':21, 'Cu2p':22, 'Cd':23
                  }

def findAAindex(aa):
    if aa in ALL_AAS:
        return ALL_AAS.index(aa)
    else:
        return 0 #UNK

def fa2gentype(fats):
    gts = {'Nbb':'Nad','Npro':'Nad3','NH2O':'Nad','Ntrp':'Nin','Nhis':'Nim','NtrR':'Ngu2','Narg':'Ngu1','Nlys':'Nam',
           'CAbb':'CS1','CObb':'CDp','CH1':'CS1','CH2':'CS2','CH3':'CS3','COO':'CDp','CH0':'CR','aroC':'CR','CNH2':'CDp',
           'OCbb':'Oad','OOC':'Oat','OH':'Ohx','ONH2':'Oad',
           'S':'Ssl','SH1':'Sth',
           'HNbb':'HN','HS':'HS','Hpol':'HO',
           'Phos':'PG5', 'Oet2':'OG3', 'Oet3':'OG3' #Nucleic acids
    }

    gents = []
    for at in fats:
        if at in gentype2num:
            gents.append(at)
        else:
            gents.append(gts[at])
    return gents

def get_AAtype_properties(ignore_hisH=True,
                          extrapath='',
                          extrainfo={}):
    qs_aa = {}
    atypes_aa = {}
    atms_aa = {}
    bnds_aa = {}
    repsatm_aa = {}
    
    iaa = 0 #"UNK"
    for aa in AMINOACID+NUCLEICACID+METAL:
        iaa += 1
        p = defaultparams(aa)
        atms,q,atypes,bnds,repsatm = read_params(p)
        atypes_aa[iaa] = fa2gentype([atypes[atm] for atm in atms])
        qs_aa[iaa] = q
        atms_aa[iaa] = atms
        bnds_aa[iaa] = bnds
        if aa in AMINOACID:
            repsatm_aa[iaa] = atms.index('CA')
        else:
            repsatm_aa[iaa] = repsatm

    if extrapath != '':
        params = glob.glob(extrapath+'/*params')
        for p in params:
            aaname = p.split('/')[-1].replace('.params','')
            args = read_params(p,aaname=aaname)
            if not args:
                print("Failed to read extra params %s, ignore."%p)
                continue
            else:
                #print("Read %s for the extra res params for %s"%(p,aaname))
                pass
            atms,q,atypes,bnds,repsatm = args
            atypes = [atypes[atm] for atm in atms] #same atypes
            extrainfo[aaname] = (q,atypes,atms,bnds,repsatm)
    if extrainfo != {}:
        print("Extra residues read from %s: "%extrapath, list(extrainfo.keys()))
    return qs_aa, atypes_aa, atms_aa, bnds_aa, repsatm_aa
