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
    T sigf;
    T fb_f;
    T fb_p;
    T fd_f;
    T fd_p;

    HKL(int h, int k, int l, T fb_f, T fb_p, T fd_f, T fd_p, T f, T sigf) : h(h), k(k), l(l), fb_f(fb_f), fb_p(fb_p), fd_f(fd_f), fd_p(fd_p), f(f), sigf(sigf){}
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
        auto pair = std::make_pair(i.f, i.sigf);
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

        clipper::datatypes::F_phi<float> fbest_reflection = fbest[ih];
        clipper::datatypes::F_phi<float> fdiff_reflection = fdiff[ih];
        clipper::datatypes::F_sigF<float> fobs_reflection = fobs[ih];

        HKL<float> output_hkl = {hkl.h(), hkl.k(), hkl.l(), fbest_reflection.f(), fbest_reflection.phi(), fdiff_reflection.f(),
                          fdiff_reflection.phi(), fobs_reflection.f(), fobs_reflection.sigf()};
        output_hkls.push_back(output_hkl);
    }

    return output_hkls;
}


//HKLVector expand_fsigf_to_lower_symmetry(HKLVector& arr,
//                                       clipper::Spgr_descr& spg,
//                                       clipper::Cell_descr& cell,
//                                       clipper::Resolution& res
//) {
//    bool debug = false;
//    typedef clipper::HKL_info::HKL_reference_index HRI;
//
//    const clipper::Spacegroup spacegroup(spg);
//    const clipper::Cell unit_cell(cell);
//    clipper::HKL_info hkl_info = {spacegroup, unit_cell, res,
//                                  true};
//
//    std::vector<clipper::HKL> hkls;
//    hkls.reserve(arr.size());
//    std::map<std::vector<int>, std::pair<float, float>> hkl_map = {};
//    for (auto &i: arr) {
//        std::vector<int> hkl = {i.h, i.k, i.l};
//        hkls.push_back({i.h, i.k, i.l});
//        auto pair = std::make_pair(i.fb_f, i.fb_p);
//        hkl_map.insert({hkl, pair});
//    }
//
//    hkl_info.add_hkl_list(hkls);
//
//    clipper::HKL_data<clipper::data32::F_sigF> fobs(hkl_info);
//
//    for (HRI ih = fobs.first(); !ih.last(); ih.next()) {
//        auto hkl = ih.hkl();
//        auto key = {hkl.h(), hkl.k(), hkl.l()};
//        auto value = hkl_map[key];
//        clipper::data32::F_sigF fphi = {value.first, value.second};
//        fobs[ih] = fphi;
//    }
//
//    clipper::Grid_sampling grid = {spacegroup, unit_cell, res};
//      clipper::HKL_info newhkl( clipper::Spacegroup( clipper::Spgr_descr( 1 ) ),
//                                unit_cell, res, true);
//
//      clipper::HKL_data<clipper::data32::F_sigF> newdata(newhkl);
//
//        clipper::HKL_info::HKL_reference_index ih;
//        clipper::HKL_info::HKL_reference_coord ik( hkl_info, clipper::HKL() );
//        for ( ih = newhkl.first(); !ih.last(); ih.next() ) {
//            ik.set_hkl( ih.hkl() );
//            newdata[ih] = fobs[ik];
//        }
//
//    std::vector<HKL<float>> output_hkls;
//    output_hkls.reserve(hkl_info.num_reflections());
//    for (HRI ih = newdata.first(); !ih.last(); ih.next() ) {
//        clipper::HKL hkl = ih.hkl();
//
//        HKL<float> output_hkl = {hkl.h(), hkl.k(), hkl.l(), (float)newdata[hkl].f(), (float)newdata[hkl].sigf(), 0.0, 0.0};
//        output_hkls.push_back(output_hkl);
//    }
//
//    return output_hkls;
//}
//
//
//HKLVector expand_fphi_to_lower_symmetry(HKLVector& arr,
//                                         clipper::Spgr_descr& spg,
//                                         clipper::Cell_descr& cell,
//                                         clipper::Resolution& res
//) {
//    bool debug = false;
//    typedef clipper::HKL_info::HKL_reference_index HRI;
//
//    const clipper::Spacegroup spacegroup(spg);
//    const clipper::Cell unit_cell(cell);
//    clipper::HKL_info hkl_info = {spacegroup, unit_cell, res,
//                                  true};
//
//    std::vector<clipper::HKL> hkls;
//    hkls.reserve(arr.size());
//    std::map<std::vector<int>, std::pair<float, float>> hkl_map = {};
//    for (auto &i: arr) {
//        std::vector<int> hkl = {i.h, i.k, i.l};
//        hkls.push_back({i.h, i.k, i.l});
//        auto pair = std::make_pair(i.fb_f, i.fb_p);
//        hkl_map.insert({hkl, pair});
//    }
//
//    hkl_info.add_hkl_list(hkls);
//
//    clipper::HKL_data<clipper::data32::F_phi> fobs(hkl_info);
//
//    for (HRI ih = fobs.first(); !ih.last(); ih.next()) {
//        auto hkl = ih.hkl();
//        auto key = {hkl.h(), hkl.k(), hkl.l()};
//        auto value = hkl_map[key];
//        clipper::data32::F_phi fphi = {value.first, value.second};
//        fobs[ih] = fphi;
//    }
//
//    clipper::HKL_info newhkl( clipper::Spacegroup( clipper::Spgr_descr( 1 ) ),
//                              unit_cell, res, true );
//
//    clipper::HKL_data<clipper::data32::F_phi> newdata(newhkl);
//
//    for (HRI ih = newhkl.first(); !ih.last(); ih.next() ) {
//        newdata[ih] = fobs[ih.hkl()];
//    }
//
//    std::vector<HKL<float>> output_hkls;
//    for (HRI ih = newdata.first(); !ih.last(); ih.next() ) {
//        clipper::HKL hkl = ih.hkl();
//
//        HKL<float> output_hkl = {hkl.h(), hkl.k(), hkl.l(), (float)newdata[hkl].f(), (float)newdata[hkl].phi(), 0.0, 0.0};
//        output_hkls.push_back(output_hkl);
//    }
//
//    return output_hkls;
//}



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
            .def(nb::init<int, int, int, float, float, float, float, float, float>())
            .def_ro("h", &HKL<float>::h)
            .def_ro("k", &HKL<float>::k)
            .def_ro("l", &HKL<float>::l)
            .def_ro("fb_f", &HKL<float>::fb_f)
            .def_ro("fb_p", &HKL<float>::fb_p)
            .def_ro("fd_f", &HKL<float>::fd_f)
            .def_ro("fd_p", &HKL<float>::fd_p)
            .def_ro("f", &HKL<float>::f)
            .def_ro("sigf", &HKL<float>::sigf);

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

//    m.def("expand_fsigf_to_p1", expand_fsigf_to_lower_symmetry, "array"_a, "spacegroup"_a, "cell"_a, "resolution"_a);
//    m.def("expand_fphi_to_p1", expand_fphi_to_lower_symmetry, "array"_a, "spacegroup"_a, "cell"_a, "resolution"_a);

}