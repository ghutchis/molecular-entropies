from itertools import product, combinations, compress
import ast
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdmolops
import py_rdl

def GetNeighbours(mol, idx):
    """
    Get Atom Neighbours.

    Input:

    mol: rdMol

    idx: Int

    Return:

    atomidx: list

    bonds: list
    """
    connected_atoms = mol.GetAtomWithIdx(idx).GetNeighbors()
    atomidx = [atom.GetIdx() for atom in connected_atoms]
    atomnicno = [atom.GetAtomicNum() for atom in connected_atoms]
    bonds = [int(mol.GetBondBetweenAtoms(idx,x).GetBondType()) for x in atomidx]
    return atomidx, bonds

def GetShortestPathBetweenRing(mol, ring1, ring2):
    """
    Compute shortest path between two non adjacent rings

    Input:

    mol: rdmol

    ring1: list

    ring2: list
    Return:

    path: tuple
    """
    pathlength = []
    record = []
    for x in ring1:
        for y in ring2:
            record.append((x,y))
            s = rdmolops.GetShortestPath(mol,x,y)
            pathlength.append(len(s))
    if any(pathlength):
        minlength = min(pathlength)
        pos = pathlength.index(minlength)
        apair = record[pos]
        spath = rdmolops.GetShortestPath(mol,apair[0],apair[1])
    else:
        spath = []
    return spath

def PathOrder(mol,path):
    """
    Get the path order A-->B/B-->A

    Input: 

    mol: rdMol

    path: list/tuple
    """
    carbamate = Chem.MolFromSmarts("[#8][CX3](=O)[#7]")
    urea_1 = Chem.MolFromSmarts("[#7&R][CX3&R](=O)[#7&R]")
    urea_2 = Chem.MolFromSmarts("[#7&R][CX3&!R](=O)[#7&!R]")
    urea_3 = Chem.MolFromSmarts("[#7&!R]!@;-[CX3&!R](=O)[#7&!R]")
    ketone = Chem.MolFromSmarts("[!#7&!#8][CX3](=O)[!#7&!#8]")
    ester = Chem.MolFromSmarts("[OX2H0][CX3](=O)[!#7&!#8]")
    ether = Chem.MolFromSmarts("[!$([C](=O))][OX2H0][!$([C](=O))]")
    amide1 = Chem.MolFromSmarts("[#7&R]@[C&R](=O)@[!#7&!#8]")
    amide2 = Chem.MolFromSmarts("[#7&R]!@;-[C&!R](=O)[!#7&!#8]")
    amide3 = Chem.MolFromSmarts("[#7&!R]!@;-[C&!R](=O)[!#7&!#8]")
    pattern = [carbamate, urea_1, urea_2, urea_3, ketone, ester, ether, amide1, amide2, amide3]
    pattern_match = []
    pipipath = list(path)
    pathsize = len(pipipath)
    for pat in pattern:
        match = mol.GetSubstructMatches(pat)
        subpattern_match = []
        for mat in match:
            if len(set(mat).intersection(set(pipipath)))>2:
                subpattern_match.append(True)
            else:
                subpattern_match.append(False)
        pattern_match.append(any(subpattern_match))
    if sum(pattern_match)>0:
        first_index = pattern_match.index(True) # first pattern match to decide the path order
        first_pattern_match = mol.GetSubstructMatches(pattern[first_index])
        allindex = []
        for fpm in first_pattern_match:
            if (len(set(fpm).intersection(set(pipipath)))>2) and (fpm[1] in pipipath):
                allindex.append(pipipath.index(fpm[1]))
        forwardindex = min(allindex)
        backwardindex = min([pathsize-1-x for x in allindex])
        if forwardindex<backwardindex:
            pathorder = pipipath # Original Order
        else:
            pipipath.reverse()
            pathorder = pipipath # Reverse order
    else:
        pathorder = None # No pi-pi stacking
        first_index = None
    return pathorder, first_index


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

def GetAromaticRings(mol):
    """
    Identify aromatic rings in a molecule
    """
    edge = []
    aromaticrings = []
    for bond in mol.GetBonds():
        edge.append(((bond.GetBeginAtomIdx(),bond.GetEndAtomIdx())))
    data = py_rdl.Calculator.get_calculated_result(edge)
    for urf in data.urfs:
        rcs = data.get_relevant_cycles_for_urf(urf)
        for rc in rcs:
            aromatic = [mol.GetAtomWithIdx(node).GetIsAromatic() for node in rc.nodes]
            if all(aromatic):
                aromaticrings.append(rc.nodes)
    return aromaticrings

def CalcPiPiStackPathFeature(mol, path):
    """
    Calculate Path Features
    """
    atomicnum = [mol.GetAtomWithIdx(atom).GetAtomicNum() for atom in path]
    hybridization = [int(mol.GetAtomWithIdx(atom).GetHybridization()) for atom in path]
    inring = GetInRingX(mol, path)
    ringlist = GetURFNodes(mol)
    pathsize = len(path)
    ringmembership = []
    for atom in path:
        ringmembership.append(sum([atom in subring for subring in ringlist]))
    pathfeature = ["{}{}{}{}{}{}{}{}{}{}{}{}".format(*atomicnum[x:x+3],*hybridization[x:x+3],*inring[x:x+3],*ringmembership[x:x+3]) for x in range(pathsize-2)]
    return pathfeature


with open("pipistackdict.txt") as f:
    dictionary = f.read()
fraglib = ast.literal_eval(dictionary)

def FindPotentialPiPiStacking(mol):
    """
    Identify potential pi-pi stacking in molecule
    """
    aromaticrings = GetAromaticRings(mol)
    pipistack_path = []
    if len(aromaticrings)>1:
        naromaticrings = len(aromaticrings)
        idxcombination = list(combinations(range(naromaticrings),2))
        for idxpair in idxcombination:
            first, second = idxpair
            firstring, secondring = aromaticrings[first], aromaticrings[second]
            if len(set(firstring).intersection(set(secondring)))==0: # non-intersect ring pairs
                path = GetShortestPathBetweenRing(mol, firstring, secondring)
                order_path, patternid = PathOrder(mol,path)
                if order_path is not None:
                    feature = CalcPiPiStackPathFeature(mol, order_path)
                    featuresize = len(feature)
                    if featuresize<=15:
                        inlist = []
                        K = featuresize/2 if featuresize%2==0 else (featuresize+1)/2
                        for i,f in enumerate(feature):
                            if i<K:
                                inlist.append(f in fraglib.get(patternid)[i])
                            else:
                                inlist.append(f in fraglib.get(patternid)[15-(featuresize-i)])
                        if all(inlist):
                            pipistack_path.append(order_path)
    return pipistack_path

def CalcPiPiStackFoldability(mol):
    rotor = Chem.MolFromSmarts("[!$(*#*)&!D1]-!@[!$(*#*)&!D1]")
    rotormatch = [sorted(r) for r in mol.GetSubstructMatches(rotor)]
    pipistackpath = FindPotentialPiPiStacking(mol)
    rotvec = np.zeros(len(rotormatch))
    if len(pipistackpath)>0:
        for path in pipistackpath:
            pathsize = len(path)
            tmp = [rotormatch.index(sorted((path[i],path[i+1]))) for i in range(pathsize-1) if (sorted((path[i],path[i+1])) in rotormatch)]
            for position in tmp:
                rotvec[position]+=1
        foldability = sum([min(rotvec[x],1) for x in range(len(rotormatch))])
    else:
        foldability = 0
    return foldability
