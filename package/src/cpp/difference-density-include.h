#include <string>
#include <optional>

class DifferenceDensityInput { 
public:
    DifferenceDensityInput(
        const std::string& mtzin,
        const std::string& pdbin,
        const std::string& colin_fo,
        const std::string& colin_fc
    ) {
        this->mtzin = mtzin;
        this->pdbin = pdbin;

        this->colin_fo = colin_fo; 
        this->colin_fc = colin_fc; 

        if (mtzin.empty()) { throw std::runtime_error("MTZ path must not be empty");}

    };

    [[nodiscard]] std::string get_mtz_path() const { return mtzin; }

    [[nodiscard]] std::optional<std::string> get_pdb_path() const {
        if (pdbin.empty()) return std::nullopt;
        return pdbin;
    }


    [[nodiscard]] std::optional<std::string> get_fobs() const {
        if (colin_fo.empty()) {
            return std::nullopt;
        }
        return colin_fo;
    }

    [[nodiscard]] std::optional<std::string> get_fc() const {
        if (colin_fc.empty()) {
            return std::nullopt;
        }
        return colin_fc;
    }

private:
    std::string mtzin;
    std::string pdbin;
    std::string colin_fo;
    std::string colin_fc;
};

class DifferenceDensityOutput { 
public: 
    DifferenceDensityOutput(const std::string& mtzout) {
        this->mtzout = mtzout; 

        if (mtzout == "") {throw std::runtime_error("PDB Out must not be empty");}
    }; 

    [[ nodiscard ]] std::string get_mtz_out() const { return mtzout; }

private:
    std::string mtzout; 

};