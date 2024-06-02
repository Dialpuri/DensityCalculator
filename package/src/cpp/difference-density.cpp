//
// Created by Jordan Dialpuri on 16/02/2024.
//

#include "difference-density-run.h"
#include <clipper/clipper-gemmi.h>
#include <gemmi/mtz.hpp>

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/bind_vector.h>

namespace nb = nanobind;

using namespace nb::literals;


template <typename T>
struct HKL {
    int h, k, l;
    T f;
    T p;
    HKL(int h, int k, int l, T f, T p) : h(h), k(k), l(l), f(f), p(p){}
};
using HKLVector = std::vector<HKL<float>>;

void convert_bfactors_to_u_iso(std::vector<clipper::MAtom>& atom_list) {
    for (auto& atom: atom_list) {
        atom.set_u_iso(clipper::Util::b2u(atom.u_iso()));
    }
}

HKLVector calculate_difference_density(HKLVector& arr,
                                       std::vector<clipper::MAtom>& atom_list,
                                       clipper::Spgr_descr& spg,
                                       clipper::Cell_descr& cell,
                                       clipper::Resolution& res
) {
    bool debug = false;
    typedef clipper::HKL_info::HKL_reference_index HRI;

    const clipper::Spacegroup spacegroup(spg);
    const clipper::Cell unit_cell(cell);
    clipper::HKL_info hkl_info = {spacegroup, unit_cell, res,
                                  true};

    std::vector<clipper::HKL> hkls;
    hkls.reserve(arr.size());
    std::map<std::vector<int>, std::pair<float, float>> hkl_map = {};
    for (auto &i: arr) {
        std::vector<int> hkl = {i.h, i.k, i.l};
        hkls.push_back({i.h, i.k, i.l});
        auto pair = std::make_pair(i.f, i.p);
        hkl_map.insert({hkl, pair});
    }

    hkl_info.add_hkl_list(hkls);

    clipper::HKL_data<clipper::data32::F_sigF> fobs(hkl_info);
    clipper::HKL_data<clipper::data32::F_phi> fphic(hkl_info);

    for (HRI ih = fobs.first(); !ih.last(); ih.next()) {
        auto hkl = ih.hkl();
        auto key = {hkl.h(), hkl.k(), hkl.l()};
        auto value = hkl_map[key];
        clipper::data32::F_sigF fphi = {value.first, value.second};
        fobs[ih] = fphi;
    }

    clipper::Grid_sampling grid = {spacegroup, unit_cell, res};


    if (debug) {
        std::cout << "Performing calculation with " << atom_list.size() << " atoms" << std::endl;
        std::cout << "Resolition limit is " << res.limit() << std::endl;
        std::cout << "Cell is " << unit_cell.format() << std::endl;
    }

    convert_bfactors_to_u_iso(atom_list);
    clipper::Xmap<float> calculated_map = {spacegroup, unit_cell, grid};
    clipper::EDcalc_iso<float> ed_calc = {2};
    ed_calc(calculated_map, atom_list);
    calculated_map.fft_to(fphic);

    clipper::HKL_data<clipper::data32::Flag> modeflag( fobs );
    for ( HRI ih = modeflag.first(); !ih.last(); ih.next() )
        if ( !fobs[ih].missing() )
            modeflag[ih].flag() = clipper::SFweight_spline<float>::BOTH;
        else
            modeflag[ih].flag() = clipper::SFweight_spline<float>::NONE;


    clipper::HKL_data<clipper::data32::F_phi> fdiff(fobs );
    clipper::HKL_data<clipper::data32::F_phi> fbest(fobs );
    clipper::HKL_data<clipper::data32::Phi_fom> phiw ( fobs );

    clipper::SFweight_spline<float> sfw( hkl_info.num_reflections(), 20 );
    bool success = sfw(fbest, fdiff, phiw, fobs, fphic, modeflag );
    if (success && debug) std::cout << "Sigma-A calculation successful" << std::endl;

    std::vector<HKL<float>> output_hkls;
    output_hkls.reserve(hkl_info.num_reflections());
    for (HRI ih = fdiff.first(); !ih.last(); ih.next() ) {
        clipper::HKL hkl = ih.hkl();
        if (hkl.h() == 0 && hkl.l() == 0 && hkl.k() == 0) { continue;} // Don't include the 0,0,0 reflection
        HKL output_hkl = {hkl.h(), hkl.k(), hkl.l(), fdiff[ih].f(), fdiff[ih].phi()};
        output_hkls.push_back(output_hkl);
    }

    return output_hkls;
}


NB_MODULE(density_calculator, m) {
    nb::class_<DifferenceDensityInput>(m, "Input")
            .def(nb::init< const std::string&, // mtzin
             const std::string&, // pdbin
             const std::string&, // colin_fo
             const std::string&  // colin_fc
                 >(), 
             "mtzin"_a, "pdbin"_a, "colin_fo"_a, "colin_fc"_a
             );

    nb::class_<DifferenceDensityOutput>(m, "Output")
            .def(nb::init< const std::string&>()); // pdbout, xmlout

    m.def("run", &run, "input"_a, "output"_a, "Run nucleofind-build");


    nb::class_<HKL<float>>(m, "HKL")
            .def(nb::init<int, int, int, float, float>())
            .def_ro("h", &HKL<float>::h)
            .def_ro("k", &HKL<float>::k)
            .def_ro("l", &HKL<float>::l)
            .def_ro("f", &HKL<float>::f)
            .def_ro("p", &HKL<float>::p);


    nb::class_<clipper::Spgr_descr>(m, "SpaceGroup")
            .def(nb::init<const std::string&>());

    nb::class_<clipper::Cell_descr>(m, "Cell")
            .def(nb::init<double, double, double, double, double, double>());

    nb::class_<clipper::Resolution>(m, "Resolution")
        .def(nb::init<double>());

    nb::class_<clipper::Atom>(m, "Atom")
            .def(nb::init<>())
            .def("set_element", &clipper::Atom::set_element, "s"_a)
            .def("set_coord_orth", &clipper::Atom::set_coord_orth, "s"_a)
            .def("set_occupancy", &clipper::Atom::set_occupancy, "s"_a)
            .def("set_u_iso", &clipper::Atom::set_u_iso, "s"_a)
            .def("set_u_aniso_orth", &clipper::Atom::set_u_aniso_orth, "s"_a);

    nb::class_<clipper::String>(m, "String")
            .def(nb::init<std::string>());

    nb::class_<clipper::Coord_orth>(m, "Coord")
            .def(nb::init<float, float, float>());

    nb::class_<clipper::MAtom>(m, "MAtom")
            .def(nb::init<clipper::Atom&>());

    nb::class_<clipper::MMonomer>(m, "Residue")
            .def(nb::init<>())
            .def("insert", &clipper::MMonomer::insert, "atom"_a, "pos"_a)
            .def("set_id", &clipper::MMonomer::set_id, "id"_a)
            .def("set_type", &clipper::MMonomer::set_type, "type"_a);

    nb::class_<clipper::MPolymer>(m, "Chain")
            .def(nb::init<>())
            .def("insert", &clipper::MPolymer::insert, "atom"_a, "pos"_a)
            .def("set_id", &clipper::MPolymer::set_id, "atom"_a);

    nb::class_<clipper::MModel>(m, "Model")
            .def(nb::init<>())
            .def("insert", &clipper::MModel::insert, "atom"_a, "pos"_a);
    
    m.def("calculate_difference_density",
          calculate_difference_density,
          "array"_a, "atom_list"_a, "spacegroup"_a, "cell"_a, "resolution"_a);
}