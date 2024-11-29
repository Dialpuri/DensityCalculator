//
// Created by Jordan Dialpuri on 29/11/2024.
//

#include <emscripten.h>
#include <emscripten/bind.h>

#include <gemmi/model.hpp>
#include <gemmi/mmread.hpp>
#include <gemmi/cif2mtz.hpp>
#include <gemmi/cif.hpp>
#include <iostream>
#include <gemmi/to_cif.hpp>    // cif::Document -> file
#include <gemmi/to_mmcif.hpp>

#include <clipper/clipper-ccp4.h>
#include <clipper/clipper-contrib.h>
#include <clipper/clipper-minimol.h>
#include <clipper/clipper.h>


using namespace emscripten;

gemmi::Structure parse_coordinates(const std::string& file) {
    char *c_data = (char *)file.c_str();
    size_t size = file.length();

    if (size == 0) {
        return {};
    }

    ::gemmi::Structure structure = ::gemmi::read_structure_from_char_array(c_data, size, "");
    return structure;
}

struct SizedString {
      SizedString(std::string& str, size_t size): str(str), size(size) {}
      std::string str;
      size_t size;
  };

SizedString generateMTZ(const std::string& coordinate_file, const std::string& cif_file) {
    gemmi::Structure structure = parse_coordinates(coordinate_file);
    std::ofstream os("structure.cif");
    gemmi::cif::write_cif_to_stream(os, gemmi::make_mmcif_document(structure));
    os.close();

    auto rblocks = gemmi::as_refln_blocks(gemmi::cif::read_string(cif_file).blocks);
    std::ostringstream out;
    gemmi::CifToMtz c2m;
    c2m.verbose = true;
    gemmi::Mtz mtz = c2m.convert_block_to_mtz(rblocks[0], out);
    mtz.write_to_file("reflections.mtz");

    typedef clipper::HKL_info::HKL_reference_index HRI;

    clipper::CCP4MTZfile mtzfile;
    mtzfile.set_column_label_mode(clipper::CCP4MTZfile::Legacy);

    clipper::HKL_info hkls;
    mtzfile.open_read("reflections.mtz");

    // double res = clipper::Util::max(mtzfile.resolution().limit(), 2.0);
    auto resol = clipper::Resolution(mtzfile.resolution().limit());

    hkls.init(mtzfile.spacegroup(), mtzfile.cell(), resol, true);
    clipper::HKL_data<clipper::data32::F_sigF> fobs(hkls);
    clipper::HKL_data<clipper::data32::F_phi> fphic(hkls);

    mtzfile.import_hkl_data(fobs, "FP,SIGFP");
    mtzfile.close_read();

    clipper::Spacegroup cspg = hkls.spacegroup();
    clipper::Cell cxtl = hkls.cell();
    clipper::Grid_sampling grid(cspg, cxtl, hkls.resolution());
    clipper::Xmap<float> xwrk(cspg, cxtl, grid);

    clipper::MMDBfile mfile;
    clipper::MiniMol mol;
    mfile.read_file("structure.cif");
    mfile.import_minimol(mol);

    std::vector<clipper::MAtom> atoms;
    for (int p = 0; p < mol.size(); p++) {
        for (int m = 0; m < mol[p].size(); m++) {
            for (int a = 0; a < mol[p][m].size(); a++) {
                atoms.push_back(mol[p][m][a]);
            }
        }
    }

    clipper::Atom_list atom_list = {atoms};
    clipper::Xmap<float> calculated_map = {cspg, cxtl, grid};
    clipper::EDcalc_iso<float> ed_calc = {resol.limit()};
    ed_calc(calculated_map, atoms);
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

    clipper::SFweight_spline<float> sfw( hkls.num_reflections(), 0 );
    bool success = sfw(fbest, fdiff, phiw, fobs, fphic, modeflag );
    if (success) std::cout << "Sigma-A calculation successful" << std::endl;

    mtzfile.open_append("reflections.mtz", "/calculated.mtz");
    std::string opcol = "[*/*/FWT,*/*/PHWT]";
    mtzfile.export_hkl_data(fbest, opcol);

    opcol = "[*/*/DELFWT,*/*/PHDELWT]";
    mtzfile.export_hkl_data(fdiff, opcol);

    opcol = "[*/*/FC,*/*/PHIC]";
    mtzfile.export_hkl_data(fphic, opcol);
    mtzfile.close_append();

    gemmi::Mtz calculated_mtz = gemmi::read_mtz_file("/calculated.mtz");
    std::string mtz_string;
    calculated_mtz.write_to_string(mtz_string);
    return {mtz_string, mtz_string.size()};
}

EMSCRIPTEN_BINDINGS(sails_module)
{

  emscripten::class_<SizedString>("SizedString")
      .property("size", &SizedString::size)
      .property("str", &SizedString::str);

    function("generateMTZ", &generateMTZ);
}