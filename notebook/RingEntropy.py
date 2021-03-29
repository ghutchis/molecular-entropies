import numpy as np
import pandas as pd
import itertools
import py_rdl
from rdkit import Chem

def GetElement(mol, idxlist):
    """
    Get the atomic number of the atom in the list (idxlist)

    Input: 

    mol: rdMol

    idxlist: list (atomidx)

    Return:

    atomicnum: list (atomic number)
    """
    atomicnum = [mol.GetAtomWithIdx(node).GetAtomicNum() for node in idxlist]
    return atomicnum

def GetNeighboursDetail(mol, idx):
    """
    mol: rdMol

    idx: Int
    """
    connected_atoms = mol.GetAtomWithIdx(idx).GetNeighbors()
    atomidx = [atom.GetIdx() for atom in connected_atoms]
    atomnicno = [atom.GetAtomicNum() for atom in connected_atoms]
    bonds = [int(mol.GetBondBetweenAtoms(idx,x).GetBondType()) for x in atomidx]
    return atomidx, bonds

def GetBonds(mol, idxlist):
    """
    mol

    idxlist
    """
    bonds = []
    size = len(idxlist)-1
    for i in range(size):
        atom_a, atom_b = idxlist[i], idxlist[i+1]
        bonds.append(int(mol.GetBondBetweenAtoms(atom_a,atom_b).GetBondType()))
    bonds.append(int(mol.GetBondBetweenAtoms(idxlist[0],idxlist[-1]).GetBondType()))
    return bonds

def Rearrangement(mol, idxlist):
    """
    Rearrange atom order

    Input:

    mol: rdmol

    idxlist: list (ring atom index)

    Return

    ringloop: list 
    """
    original_arrangement = idxlist
    ringsize = len(idxlist)
    end = ringsize-1
    ringloop = [idxlist[0]]
    bondorder = []
    for i in range(end):
        atomidx, bonds = GetNeighboursDetail(mol, ringloop[i])
        checklist = list(filter(lambda x: x in idxlist and x not in ringloop, atomidx))
        nextatom  = checklist[0]
        ringloop.append(nextatom)
    return ringloop

def EnumerateRingBond(mol, idxlist):
    """
    Enumerate bonds in a cycle

    Input:

    mol: rdMol

    idxlist: list (ring atoms)

    Return:

    ringbond: list of tuples [(atom_1, atom_2)]
    
    """
    ringbond = []
    size = len(idxlist)
    ringbond = [(idxlist[i%size],idxlist[(i+1)%size]) for i in range(size)]
    return ringbond

def GetShareSingleBondAtomsInRing(mol):
    """
    Identify single bond shared by multiple ring systems

    Return:

    """
    edgelist = []
    for bond in mol.GetBonds():
        edgelist.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
    data = py_rdl.Calculator.get_calculated_result(edgelist)
    rings = []
    for urf in data.urfs:
        rcs = data.get_relevant_cycles_for_urf(urf)
        for rc in rcs:
            rcs = data.get_relevant_cycles_for_urf(urf)
            ringatoms = Rearrangement(mol,list(rc.nodes))
            rings.append([str(rc.urf),ringatoms])
    NumRCs = len(rings)
    # identify the atoms/single bonds that share in fused ring
    fused_atoms = [] 
    spiro_atoms = []
    bridgehead_atoms = []
    bridgehead_atom_freq = []
    # Check for Fused Ring/Bridge Ring Atoms
    for i in itertools.combinations(range(NumRCs),2):  
        if rings[i[0]][0]!=rings[i[1]][0]:  # only consider unpaired URFs
            bondsetA = EnumerateRingBond(mol,rings[i[0]][1]) 
            bondsetB = EnumerateRingBond(mol,rings[i[1]][1])
            sorted_bondsetA = [tuple(sorted(ebond)) for ebond in bondsetA]
            sorted_bondsetB = [tuple(sorted(ebond)) for ebond in bondsetB]
            intersections = list(set(sorted_bondsetA).intersection(sorted_bondsetB))
            intersectionelement = [x for bondpair in intersections for x in bondpair]
            if any(intersections): 
                if len(intersections)==1:  
                    if int(mol.GetBondBetweenAtoms(intersections[0][0],intersections[0][1]).GetBondType())==1:
                        fused_atoms.append(intersections[0])
                else: 
                    bridgering_element = list(set(intersectionelement))
                    for node in bridgering_element:
                        if intersectionelement.count(node)==1:
                            bridgehead_atoms.append(node)
            ringsintersection = list(set(rings[i[0]][1]).intersection(set(rings[i[1]][1])))
            if len(ringsintersection)==1:
                spiro_atoms.append(ringsintersection[0])
    bridgehead_set = list(set(bridgehead_atoms))
    for bh in bridgehead_set:
        bh_freq = bridgehead_atoms.count(bh)
        bridgehead_atom_freq.append((bh,bh_freq))
    polycyclic_fused = []
    fused_atoms_list = set([x for bondpair in fused_atoms for x in bondpair])
    for atom in fused_atoms_list:
        urftmp = []
        for i in range(NumRCs):
            if atom in rings[i][1] and rings[i][0] not in urftmp:
                urftmp.append(rings[i][0])
        if len(urftmp)>2:
            polycyclic_fused.append((atom,len(urftmp)))
    return spiro_atoms, fused_atoms, bridgehead_set, bridgehead_atom_freq, polycyclic_fused

def CalcNumTerminalCH3(mol):
    """
    Calculate number of terminal CH3 in a molecule

    Input: 
    
    mol: rdMol

    Return:

    count: int
    """
    terminalCH3 = Chem.MolFromSmarts("[CH3]")
    match = mol.GetSubstructMatches(terminalCH3)
    count = len(match)
    return count

def GetCyclicAmide(mol):
    """
    
    Return:

    count: int
    """
    amide = Chem.MolFromSmarts("[CX3](=O)@[NH]")
    matches = mol.GetSubstructMatches(amide)
    return matches

def GetCyclicEster(mol):
    """
    
    Return:

    count: int
    """
    ester = Chem.MolFromSmarts("[CX3](=O)@[O]")
    matches = mol.GetSubstructMatches(ester)
    return matches

def GetCyclicThioamide(mol):
    """
    Return:

    count: int
    """
    thioamide = Chem.MolFromSmarts("[CX3](=[SX1])@[NH]")
    matches = mol.GetSubstructMatches(thioamide)
    return matches

def CalcTotalRingFlexibility(mol):
    """
    Ring Flexibility of a cycle: N-3-#non-single bond

    Input:

    mol: rdmol

    Return:

    output: float
    """
    edgelist = []
    for bond in mol.GetBonds():
        edgelist.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
    # Compute URFs
    data = py_rdl.Calculator.get_calculated_result(edgelist)
    spiro_atoms, fused_atoms, bridgehead_atoms, bridgehead_freq, polycyclic = GetShareSingleBondAtomsInRing(mol)
    ringflex = []
    urf_group = []
    thioamide = GetCyclicThioamide(mol)
    amide = GetCyclicAmide(mol)
    ester = GetCyclicEster(mol)
    for urf in data.urfs:
        rcs = data.get_relevant_cycles_for_urf(urf)
        for rc in rcs:
            ringatoms = Rearrangement(mol,list(rc.nodes))
            N = len(ringatoms) 
            ringbonds = GetBonds(mol, ringatoms) 
            pathbond = EnumerateRingBond(mol, ringatoms)
            sorted_pathbond = [tuple(sorted(bond)) for bond in pathbond]
            NumNonSingle = sum([b!=1 for b in ringbonds]) # number of non-single ring bond
            amidesum = len([mat for mat in amide if (mat[0] in ringatoms) and (mat[2] in ringatoms)])
            thioamidesum = len([mat for mat in thioamide if (mat[0] in ringatoms) and (mat[2] in ringatoms)])
            estersum = len([mat for mat in ester if (mat[0] in ringatoms) and (mat[2] in ringatoms)])
            functionalgroupsum = amidesum+thioamidesum+estersum
            # polycyclic penalty
            polycyclicp = 0.0
            for p in polycyclic:
                if p[0] in ringatoms:
                    polycyclicp+=float(p[1])-2
            # Bridge ring penalty
            if len(bridgehead_atoms)==0:
                 bridgeatomsp = 0.0
            else:
                bridgeatom_intersect = set(bridgehead_atoms).intersection(set(ringatoms))
                tmp = []
                for bh in bridgeatom_intersect:
                    for bhtup in bridgehead_freq:
                        if bh==bhtup[0]:
                            tmp.append(bhtup[1]-1)
                bridgeatomsp = sum(tmp)
            if len(spiro_atoms)==0:
                spirop = 0.0
            else:
                spirop = len(set(spiro_atoms).intersection(set(ringatoms)))
            if len(fused_atoms)==0:
                fusedp = 0.0
            else:
                fusedp = 0.0
                for f in fused_atoms:
                    if tuple(sorted(f)) in sorted_pathbond:
                        fusedp+=1.0
            # Ring Flexibilitiy Calculation
            if all([b==12 for b in ringbonds]): # aromatic ring 
                ringflex.append(0.0)
            else:
                ringcomplexity = bridgeatomsp + polycyclicp + spirop + fusedp
                val = max(float(N-3-NumNonSingle-ringcomplexity-functionalgroupsum),0.0)
                ringflex.append(val) 
            urf_group.append(str(rc.urf))
    dataframe = pd.DataFrame({"RingFlex":ringflex,"URFs":urf_group})
    flexibility = float(dataframe.groupby("URFs").mean().sum().values)
    return flexibility

