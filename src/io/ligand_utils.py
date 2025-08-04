import numpy as np

def get_atom_lines(mol2_file):
    with open(mol2_file,'r') as f:
        lines = f.readlines()

    atminfo = {}
    for i, ln in enumerate(lines):
        if ln.startswith('@<TRIPOS>MOLECULE'):
            cmpd = lines[i+1][:-1]
        if ln.startswith('@<TRIPOS>ATOM'):
            first_atom_idx = i+1
        if (ln.startswith('@<TRIPOS>BOND') or ln.startswith('@<TRIPOS>UNITY')) and (cmpd not in atminfo):
            last_atom_idx = i-1
            atminfo[cmpd] = lines[first_atom_idx:last_atom_idx+1]

    return atminfo


def xyz_from_mol2(mol2, centered=True):
    lines = get_atom_lines(mol2)
    atms = {}
    xyz = {}

    for key in lines:
        xyz[key] = []
        atms[key] = []

        coordinates = []
        for ln in lines[key]:
            x = ln.strip().split()
            atm = x[1]
            Rx = float(x[2])
            Ry = float(x[3])
            Rz = float(x[4])
            R = np.array([Rx,Ry,Rz])
            coordinates.append(R)
            atms[key].append(atm)
        coordinates = np.array(coordinates)
        center = np.average(coordinates,axis=0)
        if centered:
            xyz[key] = coordinates - center
        else:
            xyz[key] = coordinates
    return xyz, atms


def xyz_from_pdb(pdb, centered=True):
    atms = []
    xyz = {}
    with open(pdb,'r') as f:
        for l in f:
            if l.startswith('ATOM') or l.startswith('HETATM'):
                atm = l[12:16].strip()
                x = float(l[30:38])
                y = float(l[38:46])
                z = float(l[46:54])
                xyz[atm] = np.array([x,y,z])
                atms.append(atm)

    if centered:
        com = np.mean(list(xyz.values()), axis=0)
        for atm in xyz:
            xyz[atm] -= com

    return xyz, atms