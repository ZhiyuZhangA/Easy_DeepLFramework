#include "base.h"

namespace NN {
    std::unordered_map<std::string, size_t> supported_types;

    void initializeNNEngine() {
        // Initialize the supported types using config files
        std::string types_support_path("../configs/types-supported.config");
        std::ifstream fin(types_support_path);
        if (!fin.is_open()) {
            std::cerr << "Can't open the config files: " << types_support_path << std::endl;
            return;
        }

        std::string dataType;
        int size;
        while (fin >> dataType >> size) {
            supported_types[dataType] = size;
        }

        fin.close();
    }
}

