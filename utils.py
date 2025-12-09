from Bio.PDB import PDBParser, DSSP, is_aa
import Bio.PDB
import pandas as pd
import numpy as np
import pynmrstar
import re
import csv
import os
import fnmatch
from difflib import SequenceMatcher
import json


# this function is to reduce 3-letter abbreviation to 1-letter abbreviation
def abbr_reduce(long_abbr):
    dict_abbr_reduce = {}
    dict_abbr_reduce['ALA'] = 'A'
    dict_abbr_reduce['ARG'] = 'R'
    dict_abbr_reduce['ASN'] = 'N'
    dict_abbr_reduce['ASP'] = 'D'
    dict_abbr_reduce['CYS'] = 'C'
    dict_abbr_reduce['GLU'] = 'E'
    dict_abbr_reduce['GLN'] = 'Q'
    dict_abbr_reduce['GLY'] = 'G'
    dict_abbr_reduce['HIS'] = 'H'
    dict_abbr_reduce['ILE'] = 'I'
    dict_abbr_reduce['LEU'] = 'L'
    dict_abbr_reduce['LYS'] = 'K'
    dict_abbr_reduce['MET'] = 'M'
    dict_abbr_reduce['PHE'] = 'F'
    dict_abbr_reduce['PRO'] = 'P'
    dict_abbr_reduce['SER'] = 'S'
    dict_abbr_reduce['THR'] = 'T'
    dict_abbr_reduce['TRP'] = 'W'
    dict_abbr_reduce['TYR'] = 'Y'
    dict_abbr_reduce['VAL'] = 'V'

    list_long_abbr = list(dict_abbr_reduce.keys())

    if long_abbr in list_long_abbr:
        return (dict_abbr_reduce[long_abbr])
    else:
        return ('X')


# given a pdb filename, return the pdb id
def read_pdbid_from_filename(file_name):
    pdb_id = re.findall('([a-z0-9A-Z]+).pdb$', file_name)
    if len(pdb_id) == 0:
        return 1
    else:
        pdb_id = pdb_id[0]
        return pdb_id


# read pdb sequence from pdb file
def read_seq_from_pdb(file_name_pdb):
    parser = PDBParser(PERMISSIVE=1)
    pdb_id = file_name_pdb[-8:-4]
    structure = parser.get_structure(pdb_id, file_name_pdb)
    models = structure.get_list()
    model_0 = models[0]
    chains = model_0.get_list()
    chain_0 = chains[0]
    residues = chain_0.get_list()
    pri_seq = []
    for residue in residues:
        if is_aa(residue):
            resname = residue.get_resname()
            resname = abbr_reduce(resname)
            pri_seq.append(resname)
    return pri_seq


# read a pdb-bmrb dict from csv file
def read_pdb_bmrb_dict_from_csv(csv_file):
    dict_pdb_bmrb = {}
    with open(csv_file) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for (i, row) in enumerate(reader):
            if i > 0:
                dict_pdb_bmrb[row[0]] = row[1]
    return dict_pdb_bmrb


# find bmrb file name by its ID
def search_file_with_bmrb(bmrb_id, dir_nmrstar):
    if not os.path.exists(dir_nmrstar):
        print('where is dir of nmrstar files?')
        return 1
    list_all_files = os.listdir(dir_nmrstar)
    file_name = fnmatch.filter(list_all_files, '*{}*'.format(bmrb_id))
    if len(file_name) != 1:
        return 1
    else:
        file_name = file_name[0]
    return file_name


# read bmrb sequence from bmrb file
def read_seq_from_star(file_name_star):
    entry = pynmrstar.Entry.from_file(file_name_star)
    seq_one_letter_code = entry.get_tag('Entity.Polymer_seq_one_letter_code')
    if len(seq_one_letter_code) == 0:
        return None
    else:
        aa_seq = seq_one_letter_code[0]
        aa_seq = aa_seq.replace('\n', '')
        aa_seq = aa_seq.replace('\r', '')
        aa_seq = list(aa_seq)
    return aa_seq


# read order parameter set from bmrb file
def read_s2_into_pd_from_star(file_name_star):
    entry = pynmrstar.Entry.from_file(file_name_star, convert_data_types=True)
    s2_loops = entry.get_loops_by_category("Order_param")
    s2_loop = s2_loops[0]
    s2_set = s2_loop.get_tag(['Comp_index_ID', 'Comp_ID', 'Order_param_val'])
    pd_s2_set = pd.DataFrame.from_records(s2_set, columns=['Comp_index_ID', 'Comp_ID', 'Order_param_val'])
    return pd_s2_set


# 'H' for helix, 'E' for sheet, and 'C' for random coil
dict_2nd_structure_code = {'H': np.array([1, 0, 0]).tolist(), 'E': np.array([0, 1, 0]).tolist(),
                           'C': np.array([0, 0, 1]).tolist()}


# residue class for s2 prediction
class residue_s2:

    def __init__(self, resname):

        self.name = resname

        self.s2 = -1.0

        self.dist_var = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # distance variance among the forward and backward 3 residues

        self.phi_var = None
        self.psi_var = None

        self.state_3 = ''
        self.state_8 = ''
        self.ss_code = None

        self.acc = None

        self.concat_num = 0

    def set_s2(self, s2):
        self.s2 = s2

    def set_dist_var(self, dist_var):
        self.dist_var = dist_var

    def set_phi_var(self, phi_var):
        self.phi_var = phi_var

    def set_psi_var(self, psi_var):
        self.psi_var = psi_var

    def set_state_8(self, state):
        self.state_8 = state

    def set_state_3(self, state):
        self.state_3 = state

    def set_ss_code(self, ss_code):
        self.ss_code = ss_code

    def set_acc(self, acc):
        self.acc = acc

    def set_concat_num(self, concat_num):
        self.concat_num = concat_num

    def conv_8_3_state(self):
        dict_8_3_state = {'H': 'H', 'G': 'H', 'I': 'H', 'E': 'E', 'B': 'C', 'T': 'C', 'S': 'C', 'C': 'C', '-': 'C'}
        self.state_3 = dict_8_3_state[self.state_8]


# protein class for s2 training and test (s2 has experimental value)
class protein_s2:

    def __init__(self, pdb_id='', bmrb_id='', index_start_pdb=0, index_start_bmrb=0, length_eff=0):
        self.pdb_id = pdb_id
        self.bmrb_id = bmrb_id
        self.index_start_pdb = index_start_pdb
        self.index_start_bmrb = index_start_bmrb
        self.length_eff = length_eff
        self.pdb_seq = []  # amino acid sequence from pdb file
        self.bmrb_seq = []  # sequence from bmrb file
        self.matched_seq = []  # overlapped part of pdb_seq and bmrb_seq

    # read the overlapped part of pdb_seq and bmrb_seq
    def read_seq_train(self, pdb_file, bmrb_file):
        pdb_seq = read_seq_from_pdb(pdb_file)
        bmrb_seq = read_seq_from_star(bmrb_file)
        seq_matcher = SequenceMatcher(None, pdb_seq, bmrb_seq)
        (index_start_pdb, index_start_bmrb, length_eff) = seq_matcher.find_longest_match(0, len(pdb_seq), 0,
                                                                                         len(bmrb_seq))
        self.index_start_pdb = index_start_pdb
        self.index_start_bmrb = index_start_bmrb
        self.length_eff = length_eff

        for resname in pdb_seq:
            residue_temp = residue_s2(resname)
            self.pdb_seq.append(residue_temp)

        for resname in bmrb_seq:
            residue_temp = residue_s2(resname)
            self.bmrb_seq.append(residue_temp)

        for index in range(index_start_pdb, (index_start_pdb + length_eff)):
            residue_temp = residue_s2(pdb_seq[index])
            self.matched_seq.append(residue_temp)

    # read the overlapped part of pdb_seq and bmrb_seq
    def read_seq(self, pdb_file, bmrb_file):
        pdb_seq = read_seq_from_pdb(pdb_file)
        bmrb_seq = read_seq_from_star(bmrb_file)
        seq_matcher = SequenceMatcher(None, pdb_seq, bmrb_seq)
        (index_start_pdb, index_start_bmrb, length_eff) = seq_matcher.find_longest_match(0, len(pdb_seq), 0,
                                                                                         len(bmrb_seq))
        self.index_start_pdb = index_start_pdb
        self.index_start_bmrb = index_start_bmrb
        self.length_eff = length_eff

        for resname in pdb_seq:
            residue_temp = residue_s2(resname)
            self.pdb_seq.append(residue_temp)

        for resname in bmrb_seq:
            residue_temp = residue_s2(resname)
            self.bmrb_seq.append(residue_temp)

        for index in range(index_start_pdb, (index_start_pdb + length_eff)):
            residue_temp = residue_s2(pdb_seq[index])
            self.matched_seq.append(residue_temp)

    # read order parameter s2 from nmrstar file
    def read_s2_from_star(self, bmrb_file):
        pd_s2 = read_s2_into_pd_from_star(bmrb_file)
        for _, row in pd_s2.iterrows():
            index = row['Comp_index_ID'] - 1
            self.bmrb_seq[index].set_s2(float(row['Order_param_val']))

    # calculate three state secondary structure from pdb file
    def cal_ss_from_pdb(self, pdb_id, pdb_file):
        parser = PDBParser(PERMISSIVE=1)
        structure = parser.get_structure(pdb_id, pdb_file)
        model = structure[0]
        dssp = DSSP(model, pdb_file)
        ss_ = []
        for row in dssp:
            ss_.append(row[2])
        ss_.append('C')
        for index_residue in range(len(self.pdb_seq)):
            self.pdb_seq[index_residue].set_state_8(ss_[index_residue])
            self.pdb_seq[index_residue].conv_8_3_state()

    # calculate solvent-accessible surface area
    def get_rsa(self, pdb_id, pdb_file):
        parser = PDBParser(PERMISSIVE=1)
        structure = parser.get_structure(pdb_id, pdb_file)
        model = structure[0]
        dssp = DSSP(model, pdb_file)
        rsa = []
        for row in dssp:
            rsa.append(row[3])
        rsa.append(1)
        for index_residue in range(len(self.pdb_seq)):
            self.pdb_seq[index_residue].set_acc(rsa[index_residue])

    # calculate contact number, cutoff distance is 12
    def read_concat_num_from_pdb(self, pdb_id, pdb_file):
        parser = PDBParser(PERMISSIVE=1)
        structure = parser.get_structure(pdb_id, pdb_file)
        models = [structure.get_list()[0]]
        num_models = len(models)
        num_residues = len(self.pdb_seq)
        concat_num = np.full((num_residues, num_models), -1)

        for index_model, model in enumerate(models):
            chains = model.get_list()
            chain_0 = chains[0]
            residues = chain_0.get_list()
            cordi_CA = []
            for index_residue in range(num_residues):
                cordi_CA.append(residues[index_residue]['CA'])
            for n, i in enumerate(cordi_CA):
                for j in cordi_CA:
                    if abs(i - j) <= 12:
                        concat_num[n, index_model] += 1
        concat_num2 = np.mean(concat_num, axis=1)
        concat_num2 = list(concat_num2)
        concat_num3 = [0] * len(concat_num2)
        for i in range(1, len(concat_num2) - 1):
            concat_num3[i] = (concat_num2[i - 1] + concat_num2[i] + concat_num2[i + 1]) / 3
        concat_num3[0] = concat_num2[0]
        concat_num3[-1] = concat_num2[-1]
        concat_num3 = np.array(concat_num3)
        concat_num4 = concat_num3[:, np.newaxis]

        for index_residue in range(len(self.pdb_seq)):
            self.pdb_seq[index_residue].concat_num = concat_num4[index_residue][0]
        return concat_num3

    # read distance variance among N from pdb file
    def read_dist_var_from_pdb(self, pdb_id, pdb_file):
        parser = PDBParser(PERMISSIVE=1)
        structure = parser.get_structure(pdb_id, pdb_file)
        models = structure.get_list()
        num_models = len(models)
        if num_models == 1:
            return None
        num_residues = len(self.pdb_seq)
        num_dist = len(self.pdb_seq[0].dist_var)
        distance = np.full((num_residues, num_dist, num_models), -1.0)

        for index_model, model in enumerate(models):
            chains = model.get_list()
            chain_0 = chains[0]

            residues = []
            for residue in chain_0:
                if is_aa(residue):
                    residues.append(residue)
            for index_residue, residue in enumerate(residues):
                for index_dist in range(num_dist // 2):
                    # before the pilot residue
                    index_target = index_residue - num_dist // 2 + index_dist
                    if index_target >= 0:
                        CA_1 = residues[index_residue]['N']
                        CA_2 = residues[index_target]['N']
                        distance[index_residue, index_dist, index_model] = CA_1 - CA_2
                    # after the pilot residue
                    index_target = index_residue + index_dist + 1
                    if index_target < num_residues:
                        CA_1 = residues[index_residue]['N']
                        CA_2 = residues[index_target]['N']
                        distance[index_residue, index_dist + num_dist // 2, index_model] = CA_1 - CA_2

        distance_var = np.var(distance, axis=2)
        distance_var[distance_var == 0.0] = 1.0

        for index_residue in range(len(self.pdb_seq)):
            self.pdb_seq[index_residue].dist_var = distance_var[index_residue]

        return distance, distance_var

    # read torsion angles phi and psi variance from pdb file
    def read_torsion_var_from_pdb(self, pdb_id, pdb_file):
        parser = PDBParser(PERMISSIVE=1)
        structure = parser.get_structure(pdb_id, pdb_file)
        models = structure.get_list()
        num_models = len(models)
        num_residues = len(self.pdb_seq)
        num_angles = 2  # phi and psi
        # phi index_angle = 0, psi index_angle = 1
        torsion_angle = np.full((num_residues, num_angles, num_models), 0.0)
        model_list = []

        for index_model, model in enumerate(models):
            residue_list = []

            chains = model.get_list()
            chain = chains[0]
            for residue in chain:
                if is_aa(residue):
                    dict_atom = {}
                    for atom in residue:
                        if atom.get_name() == 'N':
                            dict_atom['N'] = atom
                        if atom.get_name() == 'CA':
                            dict_atom['CA'] = atom
                        if atom.get_name() == 'C':
                            dict_atom['C'] = atom

                    residue_list.append(dict_atom)
            model_list.append(residue_list)

        for index_residue in range(num_residues):
            for index_model in range(num_models):
                if index_residue == 0:
                    vector_N1 = model_list[index_model][index_residue]['N'].get_vector()
                    vector_CA1 = model_list[index_model][index_residue]['CA'].get_vector()
                    vector_C1 = model_list[index_model][index_residue]['C'].get_vector()
                    vector_N2 = model_list[index_model][index_residue + 1]['N'].get_vector()
                    # psi = torsion(N-CA-C-N)
                    psi_temp = Bio.PDB.calc_dihedral(vector_N1, vector_CA1, vector_C1, vector_N2)
                    torsion_angle[index_residue, 1, index_model] = psi_temp

                elif index_residue == num_residues - 1:
                    vector_C1 = model_list[index_model][index_residue - 1]['C'].get_vector()
                    vector_N2 = model_list[index_model][index_residue]['N'].get_vector()
                    vector_CA2 = model_list[index_model][index_residue]['CA'].get_vector()
                    vector_C2 = model_list[index_model][index_residue]['C'].get_vector()
                    # phi = torsion(C, N, CA, C)
                    phi_temp = Bio.PDB.calc_dihedral(vector_C1, vector_N2, vector_CA2, vector_C2)
                    torsion_angle[index_residue, 0, index_model] = phi_temp

                else:
                    vector_C0 = model_list[index_model][index_residue - 1]['C'].get_vector()
                    vector_N1 = model_list[index_model][index_residue]['N'].get_vector()
                    vector_CA1 = model_list[index_model][index_residue]['CA'].get_vector()
                    vector_C1 = model_list[index_model][index_residue]['C'].get_vector()
                    vector_N2 = model_list[index_model][index_residue + 1]['N'].get_vector()
                    phi_temp = Bio.PDB.calc_dihedral(vector_C0, vector_N1, vector_CA1, vector_C1)
                    psi_temp = Bio.PDB.calc_dihedral(vector_N1, vector_CA1, vector_C1, vector_N2)
                    torsion_angle[index_residue, 0, index_model] = phi_temp
                    torsion_angle[index_residue, 1, index_model] = psi_temp

        torsion_var = np.var(torsion_angle, axis=2)
        torsion_var[torsion_var == 0.0] = 1

        for index_residue in range(len(self.pdb_seq)):
            self.pdb_seq[index_residue].phi_var = torsion_var[index_residue, 0]
            self.pdb_seq[index_residue].psi_var = torsion_var[index_residue, 1]

    def merge(self, pdb_file, bmrb_file):
        pdb_seq = read_seq_from_pdb(pdb_file)
        bmrb_seq = read_seq_from_star(bmrb_file)
        s = SequenceMatcher(None, bmrb_seq, pdb_seq)
        seq_blocks = s.get_matching_blocks()
        for block in seq_blocks:
            (begin_seq_bmrb, begin_seq_pdb, size) = block
            if size != 0:
                for k in range(size):
                    self.pdb_seq[begin_seq_pdb + k].set_s2(self.bmrb_seq[begin_seq_bmrb + k].s2)

    # generate the matched sequence
    def merge_seq(self):
        for index in range(self.length_eff):
            self.matched_seq[index].dist_var = self.pdb_seq[index + self.index_start_pdb].dist_var

            self.matched_seq[index].phi_var = self.pdb_seq[index + self.index_start_pdb].phi_var
            self.matched_seq[index].psi_var = self.pdb_seq[index + self.index_start_pdb].psi_var

            self.matched_seq[index].state_3 = self.pdb_seq[index + self.index_start_pdb].state_3

            self.matched_seq[index].acc = self.pdb_seq[index + self.index_start_pdb].acc

            self.matched_seq[index].concat_num = self.pdb_seq[index + self.index_start_pdb].concat_num

            self.matched_seq[index].s2 = self.bmrb_seq[index + self.index_start_bmrb].s2

    # to generate a 2D numpy array
    def to_numpy(self):
        list_data = []
        for residue in self.matched_seq:
            if residue.s2 > -0.5:
                res_data = dict_2nd_structure_code[residue.state_3] + \
                           [residue.phi_var, residue.psi_var,
                            residue.dist_var[0], residue.dist_var[1],
                            residue.dist_var[2], residue.dist_var[3],
                            residue.dist_var[4], residue.dist_var[5],
                            residue.concat_num, residue.acc, residue.s2]
                list_data.append(res_data)
        array_data = np.array(list_data)
        return array_data


# protein class for s2 prediction (s2 has no experimental value)
class protein_s2_pre:

    def __init__(self, pdb_id='', length_eff=0):
        self.pdb_id = pdb_id
        self.length_eff = length_eff
        self.pdb_seq = []  # amino acid sequence from pdb file

    # read the overlapped part of pdb_seq and bmrb_seq
    def read_seq(self, pdb_file):
        pdb_seq = read_seq_from_pdb(pdb_file)
        length_eff = len(pdb_seq)
        self.length_eff = length_eff

        for resname in pdb_seq:
            residue_temp = residue_s2(resname)
            self.pdb_seq.append(residue_temp)

    # calculate three state secondary structure from pdb file
    def cal_ss_from_pdb(self, pdb_id, pdb_file):
        parser = PDBParser(PERMISSIVE=1)
        structure = parser.get_structure(pdb_id, pdb_file)
        model = structure[0]
        dssp = DSSP(model, pdb_file)
        ss_ = []
        for row in dssp:
            ss_.append(row[2])
        ss_.append('C')
        for index_residue in range(len(self.pdb_seq)):
            self.pdb_seq[index_residue].set_state_8(ss_[index_residue])
            self.pdb_seq[index_residue].conv_8_3_state()

    # calculate solvent-accessible surface area
    def get_rsa(self, pdb_id, pdb_file):
        parser = PDBParser(PERMISSIVE=1)
        structure = parser.get_structure(pdb_id, pdb_file)
        model = structure[0]
        dssp = DSSP(model, pdb_file)
        rsa = []
        for row in dssp:
            rsa.append(row[3])
        rsa.append(1)
        for index_residue in range(len(self.pdb_seq)):
            self.pdb_seq[index_residue].set_acc(rsa[index_residue])

    # calculate contact number, cutoff distance is 12
    def read_concat_num_from_pdb(self, pdb_id, pdb_file):
        parser = PDBParser(PERMISSIVE=1)
        structure = parser.get_structure(pdb_id, pdb_file)
        models = [structure.get_list()[0]]
        num_models = len(models)
        num_residues = len(self.pdb_seq)
        concat_num = np.full((num_residues, num_models), -1)

        for index_model, model in enumerate(models):
            chains = model.get_list()
            chain_0 = chains[0]
            residues = chain_0.get_list()
            cordi_CA = []
            for index_residue in range(num_residues):
                cordi_CA.append(residues[index_residue]['CA'])
            for n, i in enumerate(cordi_CA):
                for j in cordi_CA:
                    if abs(i - j) <= 12:
                        concat_num[n, index_model] += 1
        concat_num2 = np.mean(concat_num, axis=1)
        concat_num2 = list(concat_num2)
        concat_num3 = [0] * len(concat_num2)
        for i in range(1, len(concat_num2) - 1):
            concat_num3[i] = (concat_num2[i - 1] + concat_num2[i] + concat_num2[i + 1]) / 3
        concat_num3[0] = concat_num2[0]
        concat_num3[-1] = concat_num2[-1]
        concat_num3 = np.array(concat_num3)
        concat_num4 = concat_num3[:, np.newaxis]

        for index_residue in range(len(self.pdb_seq)):
            self.pdb_seq[index_residue].concat_num = concat_num4[index_residue][0]
        return concat_num3

    # read distance variance among N from pdb file
    def read_dist_var_from_pdb(self, pdb_id, pdb_file):
        parser = PDBParser(PERMISSIVE=1)
        structure = parser.get_structure(pdb_id, pdb_file)
        models = structure.get_list()
        num_models = len(models)
        if num_models == 1:
            return None
        num_residues = len(self.pdb_seq)
        num_dist = len(self.pdb_seq[0].dist_var)
        distance = np.full((num_residues, num_dist, num_models), -1.0)

        for index_model, model in enumerate(models):
            chains = model.get_list()
            chain_0 = chains[0]

            residues = []
            for residue in chain_0:
                if is_aa(residue):
                    residues.append(residue)
            for index_residue, residue in enumerate(residues):
                for index_dist in range(num_dist // 2):
                    # before the pilot residue
                    index_target = index_residue - num_dist // 2 + index_dist
                    if index_target >= 0:
                        CA_1 = residues[index_residue]['N']
                        CA_2 = residues[index_target]['N']
                        distance[index_residue, index_dist, index_model] = CA_1 - CA_2
                    # after the pilot residue
                    index_target = index_residue + index_dist + 1
                    if index_target < num_residues:
                        CA_1 = residues[index_residue]['N']
                        CA_2 = residues[index_target]['N']
                        distance[index_residue, index_dist + num_dist // 2, index_model] = CA_1 - CA_2

        distance_var = np.var(distance, axis=2)
        distance_var[distance_var == 0.0] = 1.0

        for index_residue in range(len(self.pdb_seq)):
            self.pdb_seq[index_residue].dist_var = distance_var[index_residue]

        return distance, distance_var

    # read torsion angles phi and psi variance from pdb file
    def read_torsion_var_from_pdb(self, pdb_id, pdb_file):
        parser = PDBParser(PERMISSIVE=1)
        structure = parser.get_structure(pdb_id, pdb_file)
        models = structure.get_list()
        num_models = len(models)
        num_residues = len(self.pdb_seq)
        num_angles = 2  # phi and psi
        # phi index_angle = 0, psi index_angle = 1
        torsion_angle = np.full((num_residues, num_angles, num_models), 0.0)
        model_list = []

        for index_model, model in enumerate(models):
            residue_list = []

            chains = model.get_list()
            chain = chains[0]
            for residue in chain:
                if is_aa(residue):
                    dict_atom = {}
                    for atom in residue:
                        if atom.get_name() == 'N':
                            dict_atom['N'] = atom
                        if atom.get_name() == 'CA':
                            dict_atom['CA'] = atom
                        if atom.get_name() == 'C':
                            dict_atom['C'] = atom

                    residue_list.append(dict_atom)
            model_list.append(residue_list)

        for index_residue in range(num_residues):
            for index_model in range(num_models):
                if index_residue == 0:
                    vector_N1 = model_list[index_model][index_residue]['N'].get_vector()
                    vector_CA1 = model_list[index_model][index_residue]['CA'].get_vector()
                    vector_C1 = model_list[index_model][index_residue]['C'].get_vector()
                    vector_N2 = model_list[index_model][index_residue + 1]['N'].get_vector()
                    # psi = torsion(N-CA-C-N)
                    psi_temp = Bio.PDB.calc_dihedral(vector_N1, vector_CA1, vector_C1, vector_N2)
                    torsion_angle[index_residue, 1, index_model] = psi_temp

                elif index_residue == num_residues - 1:
                    vector_C1 = model_list[index_model][index_residue - 1]['C'].get_vector()
                    vector_N2 = model_list[index_model][index_residue]['N'].get_vector()
                    vector_CA2 = model_list[index_model][index_residue]['CA'].get_vector()
                    vector_C2 = model_list[index_model][index_residue]['C'].get_vector()
                    # phi = torsion(C, N, CA, C)
                    phi_temp = Bio.PDB.calc_dihedral(vector_C1, vector_N2, vector_CA2, vector_C2)
                    torsion_angle[index_residue, 0, index_model] = phi_temp

                else:
                    vector_C0 = model_list[index_model][index_residue - 1]['C'].get_vector()
                    vector_N1 = model_list[index_model][index_residue]['N'].get_vector()
                    vector_CA1 = model_list[index_model][index_residue]['CA'].get_vector()
                    vector_C1 = model_list[index_model][index_residue]['C'].get_vector()
                    vector_N2 = model_list[index_model][index_residue + 1]['N'].get_vector()
                    phi_temp = Bio.PDB.calc_dihedral(vector_C0, vector_N1, vector_CA1, vector_C1)
                    psi_temp = Bio.PDB.calc_dihedral(vector_N1, vector_CA1, vector_C1, vector_N2)
                    torsion_angle[index_residue, 0, index_model] = phi_temp
                    torsion_angle[index_residue, 1, index_model] = psi_temp

        torsion_var = np.var(torsion_angle, axis=2)
        torsion_var[torsion_var == 0.0] = 1

        for index_residue in range(len(self.pdb_seq)):
            self.pdb_seq[index_residue].phi_var = torsion_var[index_residue, 0]
            self.pdb_seq[index_residue].psi_var = torsion_var[index_residue, 1]


    # to generate a 2D numpy array
    def to_numpy(self, pdb_id, label_file_pseudo):
        with open(label_file_pseudo + pdb_id + '.json') as obj:
            pseudo_label = json.load(obj)
        list_data = []
        for i, res in enumerate(self.pdb_seq):
            if i < 3 or i > self.length_eff - 4:
                flag = 1
            else:
                flag = 0
            feature = [flag] + dict_2nd_structure_code[res.state_3] + \
                      [res.phi_var, res.psi_var, res.dist_var[0], res.dist_var[1], res.dist_var[2],
                       res.dist_var[3],
                       res.dist_var[4], res.dist_var[5], res.concat_num, res.acc] + [pseudo_label[i]]
            list_data.append(feature)
        array_data = np.array(list_data)
        return array_data

