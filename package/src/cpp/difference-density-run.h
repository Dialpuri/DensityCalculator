#include <clipper/clipper-ccp4.h>
#include <clipper/clipper-contrib.h>
#include <clipper/clipper-minimol.h>
#include <clipper/clipper.h>

#include <algorithm>
#include "difference-density-include.h"

void run(DifferenceDensityInput& input, DifferenceDensityOutput &output) {
    typedef clipper::HKL_info::HKL_reference_index HRI;

    clipper::CCP4MTZfile mtzfile;
    mtzfile.set_column_label_mode(clipper::CCP4MTZfile::Legacy);

    clipper::HKL_info hkls;
    mtzfile.open_read(input.get_mtz_path());

    double res = clipper::Util::max(mtzfile.resolution().limit(), 2.0);
    auto resol = clipper::Resolution(res);

    hkls.init(mtzfile.spacegroup(), mtzfile.cell(), resol, true);
    clipper::HKL_data<clipper::data32::F_sigF> fobs(hkls);
    clipper::HKL_data<clipper::data32::F_phi> fphic(hkls);

    if (input.get_fobs().has_value()) mtzfile.import_hkl_data(fobs, input.get_fobs().value());
    mtzfile.close_read();

    clipper::Spacegroup cspg = hkls.spacegroup();
    clipper::Cell cxtl = hkls.cell();
    clipper::Grid_sampling grid(cspg, cxtl, hkls.resolution());
    clipper::Xmap<float> xwrk(cspg, cxtl, grid);

    clipper::MMDBfile mfile;
    clipper::MiniMol mol;
    mfile.read_file(input.get_pdb_path().value());
    mfile.import_minimol(mol);

    std::vector<clipper::MAtom> atoms;
    for (int p = 0; p < mol.size(); p++) {
        for (int m = 0; m < mol[p].size(); m++) {
            for (int a = 0; a < mol[p][m].size(); a++) {
                if (p == 0)
                    std::cout << mol[p][m][a].u_iso() << std::endl;
                atoms.push_back(mol[p][m][a]);
            }
        }
    }

    clipper::Atom_list atom_list = {atoms};
    clipper::Xmap<float> calculated_map = {cspg, cxtl, grid};
    clipper::EDcalc_iso<float> ed_calc = {res};
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

    clipper::SFweight_spline<float> sfw( hkls.num_reflections(), 20 );
    bool success = sfw(fbest, fdiff, phiw, fobs, fphic, modeflag );
    if (success) std::cout << "Sigma-A calculation successful" << std::endl;

    mtzfile.open_append(input.get_mtz_path(), "diff.mtz");
    std::string opcol = mtzfile.assigned_paths()[0].notail() + "/" + "FB";
    mtzfile.export_hkl_data(fbest, opcol);

    opcol = mtzfile.assigned_paths()[0].notail() + "/" + "DEL";
    mtzfile.export_hkl_data(fdiff, opcol);

    opcol = mtzfile.assigned_paths()[0].notail() + "/" + "FC";
    mtzfile.export_hkl_data(fphic, opcol);
    mtzfile.close_append();

    std::cout << "NORMAL TERMINATION" << std::endl;
}