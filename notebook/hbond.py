import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdmolops
import py_rdl
import ast

with open("hbonddict.txt","r") as f:
    dictionary = f.read()
hbonddict = ast.literal_eval(dictionary)

def GetURFNodes(mol):
    edge = []
    ringlist = []
    for bond in mol.GetBonds():
        edge.append(((bond.GetBeginAtomIdx(),bond.GetEndAtomIdx())))
    data = py_rdl.Calculator.get_calculated_result(edge)
    for urf in data.urfs:
        tmp = []
        rcs = data.get_relevant_cycles_for_urf(urf)
        for rc in rcs:
            for node in rc.nodes:
                if node not in tmp:
                    tmp.append(node)
        ringlist.append(tmp)
    return ringlist


def GetInRingX(mol, path):
    ringlist = GetURFNodes(mol)
    ir = []
    inring = []
    for atom in path:
        if mol.GetAtomWithIdx(atom).IsInRing():
            ir.append(1)
            inring.append(tuple([idx+1 for idx, ring in enumerate(ringlist) if atom in ring]))
        else:
            ir.append(0)
            inring.append([0])
    inringX = []
    record = []
    ridx = 1
    for i, l in enumerate(inring):
        if ir[i]==1:
            if len(record)==0:
                record.append(l)
                inringX.append(ridx)
            else:
                intersection = set(record[-1]).intersection(set(l))
                if any(intersection):
                    inringX.append(ridx)
                    record.append(tuple(intersection))
                else:
                    record.append(l)
                    ridx+=1
                    inringX.append(ridx)
        else:
            inringX.append(0)
    return inringX


def CalcHBondPathFeature(mol,path):
    """
    XXX
    """
    atomicnum = [mol.GetAtomWithIdx(atom).GetAtomicNum() for atom in path]
    hybridization = [int(mol.GetAtomWithIdx(atom).GetHybridization()) for atom in path]
    inringx = GetInRingX(mol, path)
    pathsize = len(path)
    ringlist = GetURFNodes(mol)
    ringmembership = []
    for p in path:
        ringmembership.append(sum([p in subring for subring in ringlist]))
    pathfeature = ["{}{}{}{}{}{}{}{}{}{}{}{}".format(*atomicnum[i:i+3],*hybridization[i:i+3], *inringx[i:i+3], *ringmembership[i:i+3]) for i in range(pathsize-2)]
    return pathfeature


def AcceptorType(mol, atom):
    carbonyl_inring = Chem.MolFromSmarts("[CX3&R](=O)")
    carbonyl_notinring = Chem.MolFromSmarts("[CX3&!R](=O)")
    hydroxyl = Chem.MolFromSmarts("[OX2H1]")
    alkoxy = Chem.MolFromSmarts("[OX2H0]")
    cyclicn = Chem.MolFromSmarts("[$([N,n])&R]")
    acyclicn = Chem.MolFromSmarts("[$([N,n])&R0]")
    sulfone = Chem.MolFromSmarts("[S](=O)(=O)")
    pattern = [carbonyl_inring, carbonyl_notinring, hydroxyl, alkoxy, cyclicn, acyclicn, sulfone]
    matches = [mol.GetSubstructMatches(pat) for pat in pattern]
    acceptortype = []
    for idx, mat in enumerate(matches):
        tmp = []
        for sub in mat:
            if atom in sub:
                tmp.append(True)
        acceptortype.append(any(tmp))
    if any(acceptortype):
        acceptorindex = acceptortype.index(True)
    else:
        acceptorindex = None
    return acceptorindex


def FindHBond(mol):
    donor = Chem.MolFromSmarts("[$([N;!H0;v3,v4&+1]),$([O,S;H1;+0]),n&H1&+0]") 
    acceptor = Chem.MolFromSmarts("[$([O,S;H1;v2;!$(*-*=[O,N,P,S])]),$([O,S;H0;v2]),$([O,S;-]),$([N;v3;!$(N-*=[O,N,P,S])]),n&H0&+0,$([o,s;+0;!$([o,s]:n);!$([o,s]:c:n)])]")    
    donormatch = [datom for mat in mol.GetSubstructMatches(donor) for datom in mat]
    acceptormatch = [aatom for mat in mol.GetSubstructMatches(acceptor) for aatom in mat]
    acceptor_type = [AcceptorType(mol, atom) for atom in acceptormatch]
    hbondpath = []
    for idx, aatom in enumerate(acceptormatch):
        acc_type = acceptor_type[idx]
        if acc_type in range(6) : 
            for datom in donormatch:
                tmp = []
                if aatom!=datom:
                    path = Chem.GetShortestPath(mol,datom, aatom)
                    if len(path)>=4 and len(path)<=12: # requie at least 4 atoms in the path, up to 12 atoms
                        pathfeature = CalcHBondPathFeature(mol,path)
                        hbondtable = hbonddict.get(acc_type)
                        pathsize = len(pathfeature)
                        K = pathsize/2 if pathsize%2==0 else (pathsize+1)/2
                        for pindx, p in enumerate(pathfeature):
                            position = pindx if pindx<K else 10-(pathsize-pindx)
                            tmp.append(p in hbondtable.get(position))
                        if all(tmp):
                            hbondpath.append(path)
    fhbondpath = []
    for hp in hbondpath:
        if hp not in fhbondpath:
            fhbondpath.append(hp)
    return fhbondpath
    

def CalcHBondFoldability(mol):
    rotor = Chem.MolFromSmarts("[!$(*#*)&!D1]-!@[!$(*#*)&!D1]")
    rotormatch = [sorted(r) for r in mol.GetSubstructMatches(rotor)]
    hbond = FindHBond(mol)
    if len(hbond)==0: # no hbond path being identified
        value = 0 
    else:
        vec = np.zeros(len(rotormatch))
        donor = []
        group = []
        for h in hbond:
            if h[0] not in donor:
                donor.append(h[0])
                group.append([h])
            else:
                group[donor.index(h[0])].append(h)
        for subgroup in group:
            tmpvec = np.zeros(len(rotormatch))
            size = len(subgroup)
            for g in subgroup:
                gsize = len(g)
                tmprecord = [rotormatch.index(sorted((g[i],g[i+1]))) for i in range(gsize-1) if sorted((g[i],g[i+1])) in rotormatch]
                for x in tmprecord:
                    tmpvec[x]+=1/size
            vec+=tmpvec
        value=0
        for v in vec:
            value+=min(v,1)
    return value
