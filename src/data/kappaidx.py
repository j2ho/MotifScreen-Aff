import numpy as np

gentype2ridx = {'CS':0,'CS1':0,'CS2':0,'CS3':0,'CST':0,'CSQ':0,'CSp':0,
                'CD':1,'CD1':1,'CD2':1,'CDp':1,
                'CT':2,'CTp':2,
                'CR':3,'CRp':3,
                'HN':4,'HO':4,'HS':4,
                'Nam':5,'Nam2':5,'NG3':5,
                'Nad':6,'Nad3':6,'Nin':6,'Nim':6,'Ngu1':6,'Ngu2':6,'NG2':6,'NG21':6,'NG22':6,
                'NG1':7,
                'Ohx':8,'OG3':8,'Oet':8,'OG31':8,
                'Oal':9, 'Oad':9, 'Oat':9, 'Ofu':9, 'Ont':9, 'OG2':9,
                'Sth':10, 'Ssl':10, 'SR':11,  'SG2':11, 'SG3':10, 'SG5':10, 'PG3':12, 'PG5':12,
                'F':13, 'Cl':14, 'Br':15, 'I':16, 'FR':13, 'ClR':14, 'BrR':15, 'IR':16,
                # fa_std type
                'CNH2':1,'CH1':0,'CH3':0,'CObb':1,'aroC':1, 'COO':1,'CH2':0, 'CAbb':0,
                'Npro':6,'Nhis':6, 'Nlys':5, 'NH2O':6,'Narg':6,'Ntrp':6,'Nbb':6, 'NtrR':6,
                'OOC':9,'OCbb':9,'ONH2':9, 'OH':8,
                'S': 10, 'SH1':10,
}

sybyl2ridx = {'C.3':0,'C.2':1,'C.1':2,'C.aro':3,'C.ar':3,'C.cat':2,
              'N.3':5,'N.2':6,'N.1':7,'N.ar':6,'N.am':6,'N.4':5,
              'O.3':8,'O.2':9,'O.co2':9,
              'S.3':10,'S.2':11,'S.o2':10,
              'P.3':12,
              'F':13,'Cl':14,'Br':15,'I':16,
              'H':4}

alpha = np.array([0.00,-0.13,-0.22, #C
                  0.0, #H
                  -0.04,-0.20,-0.29, #N
                  -0.04,-0.20, #O
                  0.35,0.22, #S
                  0.43,0.30, #P
                  -0.07,0.29,0.48,0.73 # FClBrI
        ])

def find_atype(a):
    if a in gentype2ridx:
        return gentype2ridx[a]
    if a in sybyl2ridx:
        return sybyl2ridx[a]
    return 0

def calc_Kappaidx(atypes,bnds,is_aa=False):
    # calc P1,P2 (==nbnd,nang)
    P1,P2 = 0,0
    for ib,(i,j) in enumerate(bnds):
        P1 += 1

        if ib == len(bnds)-1: continue
        for k,l in bnds[ib+1:]:
            if i in [k,l] or j in [k,l]: P2 += 1

    A = len(atypes)

    if is_aa:
        P1 -= 1
        A -= 1

    if A <= 0:
        return 0,0,0

    K1 = A*(A-1)*(A-1)/P1/P1
    K2 = (A-1)*(A-2)*(A-2)/P2/P2

    asum = sum([alpha[find_atype(a)] for a in atypes])
    Ka1 = (A+asum)*(A+asum-1)*(A+asum-1)/(P1+asum)/(P1+asum)
    Ka2 = (A+asum-1)*(A+asum-2)*(A+asum-2)/(P2+asum)/(P2+asum)

    FlexID = Ka1*Ka2/A
    # K: units should be ~natm
    # FlexID: units should be ~natm

    return K1,K2,FlexID

        