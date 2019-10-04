#!/usr/bin/env python

from __future__ import print_function

import sys
import os
import math
import glob
import gzip

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import Descriptors3D

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
    data.append(file)
    try:
        data.append(Chem.MolToSmiles(m))
    except:
        continue

    with open(path) as f:
        entropy = 0.0
        count_1 = count_2 = count_3 = count_4 = count_5 = count_6 = 0
        for line in f:
            if "ensemble entropy" in line:
                entropy = float(line.split()[7])
            if "number of unique conformers" in line:
                num_to_count = int(line.split()[7])
#                print(num_to_count)
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

        data.append(Descriptors.NumRotatableBonds(m))
        data.append(len(m.GetSubstructMatches(methyl)))
        data.append(len(m.GetSubstructMatches(amine)))
        data.append(len(m.GetSubstructMatches(hydroxy)))
        data.append(Descriptors.NumHDonors(m))
        data.append(Descriptors.NumHAcceptors(m))
        data.append(Descriptors.RingCount(m))
        data.append(Descriptors.NumAromaticRings(m))

        data.append(Descriptors.ExactMolWt(m))
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
#        data.append(Descriptors.NumBridgeheadAtoms(m))
#        data.append(Descriptors.NumSpiroAtoms(m))

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
        print(sys.argv[1], *data, sep=',')
