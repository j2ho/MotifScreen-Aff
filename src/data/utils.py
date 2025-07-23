import numpy as np
import scipy.spatial
from .types import ELEMS

def sasa_from_xyz(xyz, elems, probe_radius=1.4, n_samples=50):

    atomic_radii = {"C":  2.0,"N": 1.5,"O": 1.4,"S": 1.85,"H": 0.0, #ignore hydrogen for consistency
                    "F": 1.47,"Cl":1.75,"Br":1.85,"I": 2.0,'P': 1.8, 'Null':0.0}

    areas = []
    normareas = []
    centers = xyz
    radii = np.array([atomic_radii[e] for e in elems])
    n_atoms = len(elems)

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
        center_exp = center[None,:].repeat(n_neigh,axis=0)
        d2cen = np.sum( (center_exp - xyz[neigh]) ** 2, axis=1)
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

def read_mol2(mol2,drop_H=False):
    read_cont = 0
    qs = []
    elems = []
    xyzs = []
    bonds = []
    borders = []
    atms = []

    for l in open(mol2):
        if l.startswith('@<TRIPOS>ATOM'):
            read_cont = 1
            continue
        if l.startswith('@<TRIPOS>BOND'):
            read_cont = 2
            continue
        if l.startswith('@<TRIPOS>SUBSTRUCTURE'):
            break
        if l.startswith('@<TRIPOS>UNITY_ATOM_ATTR'):
            read_cont = 0
            continue

        words = l[:-1].split()
        if read_cont == 1:

            idx = words[0]
            if words[1].startswith('BR'): words[1] = 'Br'
            if words[1].startswith('Br') or  words[1].startswith('Cl') :
                elem = words[1][:2]
            else:
                elem = words[1][0]

            if elem == 'A' or elem == 'B' :
                elem = words[5].split('.')[0]

            if elem not in ELEMS: elem = 'Null'

            atms.append(words[1])
            elems.append(elem)
            qs.append(float(words[-1]))
            xyzs.append([float(words[2]),float(words[3]),float(words[4])])

        elif read_cont == 2:
            # if words[3] == 'du' or 'un': rint(mol2)
            bonds.append([int(words[1])-1,int(words[2])-1]) #make 0-index
            bondtypes = {'0':0,'1':1,'2':2,'3':3,'ar':3,'am':2, 'du':0, 'un':0}
            borders.append(bondtypes[words[3]])

    nneighs = [[0,0,0,0] for _ in qs]
    for i,j in bonds:
        if elems[i] in ['H','C','N','O']:
            k = ['H','C','N','O'].index(elems[i])
            nneighs[j][k] += 1.0
        if elems[j] in ['H','C','N','O']:
            l = ['H','C','N','O'].index(elems[j])
            nneighs[i][l] += 1.0

    # drop hydrogens
    if drop_H:
        nonHid = [i for i,a in enumerate(elems) if a != 'H']
    else:
        nonHid = [i for i,a in enumerate(elems)]

    borders = [b for b,ij in zip(borders,bonds) if ij[0] in nonHid and ij[1] in nonHid]
    bonds = [[nonHid.index(i),nonHid.index(j)] for i,j in bonds if i in nonHid and j in nonHid]

    return np.array(elems)[nonHid], np.array(qs)[nonHid], bonds, borders, np.array(xyzs)[nonHid], np.array(nneighs,dtype=float)[nonHid], list(np.array(atms)[nonHid])


def read_mol2_batch(mol2, tags_read=None, drop_H=True, tag_only=False):
    qs_s, elems_s, xyzs_s = {}, {}, {}
    bonds_s, borders_s = {}, {}
    atms_s, nneighs_s, atypes_s = {}, {}, {}
    tags = []

    cont = open(mol2).readlines()
    il = [i for i, l in enumerate(cont) if l.startswith('@<TRIPOS>MOLECULE')] + [len(cont)]
    ihead = np.zeros(len(cont)+1, dtype=bool)
    ihead[il] = True

    # 초기화
    qs = elems = xyzs = bonds = borders = atms = atypes = []
    tag = None
    read_cont = 0

    def save_current_molecule():
        if tag is None or (tag_only or (tags_read is not None and tag not in tags_read)):
            return
        nneighs = [[0,0,0,0] for _ in qs]
        for i, j in bonds:
            if elems[i] in ['H','C','N','O']:
                k = ['H','C','N','O'].index(elems[i])
                nneighs[j][k] += 1.0
            if elems[j] in ['H','C','N','O']:
                l = ['H','C','N','O'].index(elems[j])
                nneighs[i][l] += 1.0

        if drop_H:
            nonHid = [i for i, a in enumerate(elems) if a != 'H']
        else:
            nonHid = list(range(len(elems)))
        bonds_filt = [[nonHid.index(i), nonHid.index(j)] for i,j in bonds if i in nonHid and j in nonHid]
        borders_filt = [b for b, ij in zip(borders, bonds) if ij[0] in nonHid and ij[1] in nonHid]
        elems_s[tag]   = np.array(elems)[nonHid]
        qs_s[tag]      = np.array(qs)[nonHid]
        bonds_s[tag]   = bonds_filt
        borders_s[tag] = borders_filt
        xyzs_s[tag]    = np.array(xyzs)[nonHid]
        nneighs_s[tag] = np.array(nneighs, dtype=float)[nonHid]
        atms_s[tag]    = list(np.array(atms)[nonHid])
        atypes_s[tag]  = np.array(atypes)[nonHid]

    for i, l in enumerate(cont):
        if l.startswith('#'):
            continue

        if ihead[i]:
            save_current_molecule()

            read_cont = 3
            tag = cont[i+1].strip()
            if tag not in tags:
                tags.append(tag)
            qs, elems, xyzs, bonds, borders, atms, atypes = [], [], [], [], [], [], []
            continue

        if (not ihead[i+1] and len(l.strip()) <= 1) or tag is None:
            continue

        if read_cont == 3:
            if tags_read is None or tag in tags_read:
                read_cont = 4
            else:
                read_cont = -1
            continue

        if read_cont < 0 or tag_only:
            continue

        if l.startswith('@<TRIPOS>ATOM'):
            read_cont = 1
            continue
        elif l.startswith('@<TRIPOS>BOND'):
            read_cont = 2
            continue
        elif l.startswith('@<TRIPOS>SUBSTRUCTURE') or l.startswith('@<TRIPOS>UNITY_ATOM_ATTR'):
            read_cont = 0
            continue

        if read_cont == 1:
            words = l.strip().split()
            if len(words) < 6:
                continue
            name = words[1]
            if name.startswith('BR'):
                name = 'Br'
            if name.startswith('Br') or name.startswith('Cl'):
                elem = name[:2]
            else:
                elem = name[0]

            if elem in ['A', 'B']:
                elem = words[5].split('.')[0]
            if elem not in ELEMS:
                elem = 'Null'

            atms.append(words[1])
            atypes.append(words[5])
            elems.append(elem)
            xyzs.append([float(words[2]), float(words[3]), float(words[4])])
            qs.append(float(words[-1]) if len(words) >= 9 else 0.0)

        elif read_cont == 2:
            words = l.strip().split()
            if len(words) < 4:
                continue
            i1, i2 = int(words[1])-1, int(words[2])-1
            bondtype = {'0':0, '1':1, '2':2, '3':3, 'ar':3, 'am':2, 'du':0, 'un':0}.get(words[3], 0)
            bonds.append([i1, i2])
            borders.append(bondtype)

    save_current_molecule()

    tags_order = [tag for tag in (tags_read or tags) if tag in tags]
    if not tag_only:
        elems_s   = [elems_s[tag]   for tag in tags_order]
        qs_s      = [qs_s[tag]      for tag in tags_order]
        bonds_s   = [bonds_s[tag]   for tag in tags_order]
        borders_s = [borders_s[tag] for tag in tags_order]
        xyzs_s    = [xyzs_s[tag]    for tag in tags_order]
        nneighs_s = [nneighs_s[tag] for tag in tags_order]
        atms_s    = [atms_s[tag]    for tag in tags_order]
        atypes_s  = [atypes_s[tag]  for tag in tags_order]

    return elems_s, qs_s, bonds_s, borders_s, xyzs_s, nneighs_s, atms_s, atypes_s, tags_order


def read_mol2s_xyzonly(mol2):
    read_cont = 0
    xyzs = []
    atms = []

    for l in open(mol2):
        if l.startswith('@<TRIPOS>ATOM'):
            read_cont = 1
            xyzs.append([])
            atms.append([])
            continue
        if l.startswith('@<TRIPOS>UNITY_ATOM_ATTR'):
            read_cont = 0
            continue

        if l.startswith('@<TRIPOS>BOND'):
            read_cont = 2
            continue

        words = l[:-1].split()
        if read_cont == 1:
            is_H = (words[1][0] == 'H')
            if not is_H:
                atms[-1].append(words[1])
                xyzs[-1].append([float(words[2]),float(words[3]),float(words[4])])

    return np.array(xyzs), atms