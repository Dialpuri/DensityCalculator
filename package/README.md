# Density Calculator

Density Calculator is a library for calculating a $mFo-DFc$ maps from `gemmi` in-memory objects.

## Installation 
    
    pip install density_calculator

## Usage

#### Function Signature: 

    density_calculator.calculate(structure: gemmi.Structure, mtz: gemmi.Mtz, column_names: List[str]) -> gemmi.Mtz

#### Example Use:

    import density_calculator
    import gemmi
    
    mtz: gemmi.Mtz = gemmi.read_mtz_file("reflections.mtz")
    st: gemmi.Structure = gemmi.read_structure("model.pdb")
    
    # Calculate the difference map using density_calculator
    difference_mtz: gemmi.Mtz = density_calculator.calculate(st, mtz, ["FP", "SIGFP"])
    
    # Transform the difference_mtz into a map for calculation use
    difference_map: gemmi.FloatGrid = difference_mtz.transform_f_phi_to_map("DELFWT", "PHDELWT")
    

## Development 

    pip install --no-build-isolation --config-settings=editable.rebuild=true -Cbuild-dir=build -ve .