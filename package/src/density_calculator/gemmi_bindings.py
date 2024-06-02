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


def calculate(structure: gemmi.Structure, mtz: gemmi.Mtz, column_names: List[str]):
    fobs = mtz.get_value_sigma(*column_names)

    hkls = fobs.miller_array
    values = fobs.value_array

    result = [np.append(hkls[i], [v[0], v[1]]) for i, v in enumerate(values)]
    result = [HKL(int(h), int(k), int(l), v, s) for h,k,l,v,s in result]

    spg = SpaceGroup(mtz.spacegroup.hm)
    cell = Cell(*mtz.cell.__getstate__())
    res = Resolution(mtz.resolution_high())
    model = bind_structure(structure)
    diff = calculate_difference_density(result, model, spg, cell, res)

    diff_data = np.array([[a.h, a.k, a.l, a.f, np.rad2deg(a.p)] for a in diff])

    diff_mtz = gemmi.Mtz(with_base=True)
    diff_mtz.spacegroup = mtz.spacegroup
    diff_mtz.set_cell_for_all(mtz.cell)
    diff_mtz.add_dataset('mFo-DFc')
    diff_mtz.history = ["Difference Density Calculated By Clipper - Jordan Dialpuri, Kevin Cowtan, Marcin Wojdyr 2024"]
    diff_mtz.add_column('DELFWT', 'F')
    diff_mtz.add_column('PHDELWT', 'P')
    diff_mtz.set_data(diff_data)
    diff_mtz.ensure_asu()
    return diff_mtz


def test():
    mtz = gemmi.read_mtz_file("/Users/dialpuri/Development/difference-density/5fji/5fji-sf.mtz")
    st = gemmi.read_structure("/Users/dialpuri/Development/difference-density/5fji/5fji-pnagremoved.pdb")
    difference_mtz = calculate(st, mtz, ["FP", "SIGFP"])
    difference_mtz.write_to_file('/Users/dialpuri/Development/difference-density/5fji/output.mtz')


if __name__ == "__main__":
    test()