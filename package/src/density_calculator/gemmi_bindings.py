from typing import List
import numpy as np
import gemmi

from .density_calculator import (calculate_difference_density, HKL,
                                 SpaceGroup, Cell, Resolution, Atom,
                                 Residue, Chain, Model, String,
                                 Coord, MAtom)

def bind_structure(structure: gemmi.Structure):
    m = Model()
    for chain in structure[0]:
        c = Chain()
        c.set_id(String(chain.name))
        for residue in chain:
            r = Residue()
            r.set_id(String(str(residue.seqid)))
            r.set_type(String(residue.name))
            for atom in residue:
                a = Atom()
                a.set_element(String(atom.element.name))
                a.set_coord_orth(Coord(*atom.pos.tolist()))
                a.set_occupancy(atom.occ)
                a.set_u_iso(atom.b_iso)
                r.insert(MAtom(a), -1)
            c.insert(r, -1)
        m.insert(c, -1)

    return m


def get_atom_list(structure: gemmi.Structure):
    atom_list = []
    for chain in structure[0]:
        for residue in chain:
            for atom in residue:
                a = Atom()
                a.set_element(String(atom.element.name))
                a.set_coord_orth(Coord(*atom.pos.tolist()))
                a.set_occupancy(atom.occ)
                a.set_u_iso(atom.b_iso)
                atom_list.append(MAtom(a))
    return atom_list


def calculate(structure: gemmi.Structure, mtz: gemmi.Mtz, column_names: List[str]):
    fobs = mtz.get_value_sigma(*column_names)

    hkls = fobs.miller_array
    values = fobs.value_array

    result = [np.append(hkls[i], [v[0], v[1]]) for i, v in enumerate(values)]
    result = [HKL(int(h), int(k), int(l), v, s, 0, 0) for h, k, l, v, s in result]

    spg = SpaceGroup(mtz.spacegroup.hm)
    cell = Cell(*mtz.cell.__getstate__())
    res = Resolution(mtz.resolution_high())
    atom_list = get_atom_list(structure)

    diff = calculate_difference_density(result, atom_list, spg, cell, res)

    diff_data = np.array([[a.h, a.k, a.l, a.f[0], np.rad2deg(a.p[0]), a.f[1], np.rad2deg(a.p[1])] for a in diff])

    diff_mtz = gemmi.Mtz(with_base=True)
    diff_mtz.spacegroup = mtz.spacegroup
    diff_mtz.set_cell_for_all(mtz.cell)
    diff_mtz.add_dataset('mFo-DFc')
    diff_mtz.history = ["Difference Density Calculated By Clipper"]
    diff_mtz.add_column('DELFWT', 'F')
    diff_mtz.add_column('PHDELWT', 'P')
    diff_mtz.add_column('FWT', 'F')
    diff_mtz.add_column('PHWT', 'P')
    diff_mtz.set_data(diff_data)
    diff_mtz.ensure_asu()
    return diff_mtz
