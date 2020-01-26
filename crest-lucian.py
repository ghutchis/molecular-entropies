#!/usr/bin/env python

from __future__ import print_function

import sys
import os
import glob
import gzip
import itertools
import base64

from rdkit.Chem import Descriptors
from rdkit.Chem import Descriptors3D
from rdkit.Geometry import Point3D
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import DataStructs
from rdkit.Chem import AllChem as Chem

from openbabel import pybel

# mostly inspired from Daylight examples
methyl = Chem.MolFromSmarts("[CX4H3]")
amine = Chem.MolFromSmarts("[NX3;H2,H1;!$(NC=O)]")
hydroxy = Chem.MolFromSmarts("[OX2H]")
thiol = Chem.MolFromSmarts("[#16X2H]")

for path in glob.iglob('*/*/log_*.txt'):
    subset, name, file = path.split('/')
    base = file[4:-4]

    # read the molecule from the supplied file
    sdf_file = "{}/{}/mol_{}.sdf".format(subset, name, base)
    if not os.path.isfile(sdf_file):
        # print(base + '.sdf', " can't find sdf")
        continue

    m = Chem.MolFromMolFile(sdf_file)
    data = []
    data.append("{}/{}".format(subset,file))  # name
    try:
        data.append(Chem.MolToSmiles(m))
    except:
        continue

    # read the updated coordinates from the XYZ file
    xyz_file = sdf_file[:-4] + ".xyz"
    if not os.path.isfile(xyz_file):
        continue

    try:
        xyz_mol = next(pybel.readfile("xyz", xyz_file))

        # check for hydrogens - some of these files seem to have no protons?
        has_hydrogens = False
        for atom in xyz_mol.atoms:
            if atom.atomicnum == 1:
                has_hydrogens = True
                break
        if not has_hydrogens:
            continue # weird compound, skip this

        xyz_smiles = xyz_mol.write().split()[0]
        if '.' in xyz_smiles:
            continue # no fragments - CREST messed things up
    except StopIteration:
        continue # problem reading XYZ

    vib_file = "{}/mol_{}-vib.out".format(subset, base)
    vib = rot = tr = 0.0  # default entropies
    try:
        with open(vib_file, 'r') as f:
            found = False
            for line in f:
                if 'partition function' in line:
                    found = True

                if found and 'VIB' in line:
                    vib = float(line.split()[5]) * 4.184
                if found and 'ROT' in line:
                    rot = float(line.split()[4]) * 4.184
                if found and 'TR' in line:
                    tr = float(line.split()[4]) * 4.184
    except:
        continue
    if vib == 0.0:
        continue  # didn't find vibrational data - bad case

    with open(path) as f:
        entropy = 0.0
        count_1 = count_2 = count_3 = count_4 = count_5 = count_6 = 0
        for line in f:
            if "ensemble entropy" in line:
                entropy = float(line.split()[7])
            if "number of unique conformers" in line:
                num_to_count = int(line.split()[7])
                for c in range(num_to_count):
                    line = f.readline()
                    energy = line.split()[1]
                    if float(energy) < 1.0:
                        count_1 += 1
                    if float(energy) < 2.0:
                        count_2 += 1
                    if float(energy) < 3.0:
                        count_3 += 1
                    if float(energy) < 4.0:
                        count_4 += 1
                    if float(energy) < 5.0:
                        count_5 += 1
                    if float(energy) < 6.0:
                        count_6 += 1

    if entropy != 0.0:
        data.append(entropy)
        data.append(vib)
        data.append(rot)
        data.append(tr)

        data.append(m.GetNumAtoms())
        data.append(m.GetNumBonds())
        data.append(Descriptors.ExactMolWt(m))
        data.append(Chem.ComputeMolVolume(m))

        data.append(Descriptors.NumRotatableBonds(m))
        data.append(rdMolDescriptors.CalcNumRotatableBonds(m, strict=0))

        data.append(len(m.GetSubstructMatches(methyl)))
        data.append(len(m.GetSubstructMatches(amine)))
        data.append(len(m.GetSubstructMatches(hydroxy)))
        data.append(Descriptors.NumHDonors(m))
        data.append(Descriptors.NumHAcceptors(m))
        data.append(Descriptors.RingCount(m))
        data.append(Descriptors.NumAromaticRings(m))

        data.append(Descriptors.MaxAbsPartialCharge(m))
        data.append(Descriptors.MinAbsPartialCharge(m))
        data.append(Descriptors.MaxPartialCharge(m))
        data.append(Descriptors.MinPartialCharge(m))
        data.append(Descriptors.TPSA(m))
        data.append(Descriptors.LabuteASA(m))
        data.append(Descriptors.MolMR(m))
        data.append(Descriptors.MolLogP(m))

        data.append(Descriptors.EState_VSA1(m))
        data.append(Descriptors.EState_VSA2(m))
        data.append(Descriptors.EState_VSA3(m))
        data.append(Descriptors.EState_VSA4(m))
        data.append(Descriptors.EState_VSA5(m))

        data.append(Descriptors.HallKierAlpha(m))
        data.append(Descriptors.BertzCT(m))
        data.append(Descriptors.BalabanJ(m))
        data.append(Descriptors.Ipc(m))
        data.append(Descriptors.Kappa1(m))
        data.append(Descriptors.Kappa2(m))
        data.append(Descriptors.Kappa3(m))

        data.append(Descriptors.FractionCSP3(m))
        data.append(rdMolDescriptors.CalcNumBridgeheadAtoms(m))
        data.append(rdMolDescriptors.CalcNumSpiroAtoms(m))

        data.append(Descriptors3D.Asphericity(m))
        data.append(Descriptors3D.Eccentricity(m))
        data.append(Descriptors3D.InertialShapeFactor(m))
        data.append(Descriptors3D.RadiusOfGyration(m))
        data.append(Descriptors3D.SpherocityIndex(m))

        # number of conformers under 1, 2, 3, etc. kcal/mol
        data.append(count_1)
        data.append(count_2)
        data.append(count_3)
        data.append(count_4)
        data.append(count_5)
        data.append(count_6)
        # ecfp4 fingerprint, 4096 bit vector
        data.append(Chem.GetMorganFingerprintAsBitVect(m,2,nBits=4096).ToBase64())
        # ecfp6 fingerprint, 4096 bit vector
        data.append(Chem.GetMorganFingerprintAsBitVect(m,3,nBits=4096).ToBase64())
        print(sys.argv[1], *data, sep=',')
