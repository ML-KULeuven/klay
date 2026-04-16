#include <klay/indices.h>
#include <klay/circuit.h>

namespace klay {

void cleanup(void* data) noexcept {
  delete[] static_cast<long int*>(data);
}

std::pair<Arrays, Arrays> get_indices(Circuit& c) {
    c.remove_unused_nodes();
    c.add_root_layer();
    //print_circuit(); // Helpful for debugging small circuits

    // per layer, a vector of size the number of children 
    // (but children can count twice so this might be larger 
    // than simply the previous layer)
    Arrays indices_ndarrays;
    // per layer, a vector representing the layer
    Arrays csr_ndarrays;
    
    for (std::size_t i = 1; i < c.nb_layers(); ++i) {
        std::vector<long int> child_counts(c.layers[i].size(), 0);
        std::size_t layer_size = 0;
        std::size_t layer_len = c.layers[i].size()+1;
        for (const auto *node: c.layers[i]) {
            layer_size += node->children.size();
            child_counts[node->ix] = node->children.size();
        }

        long int* csr_data = new long int[layer_len];
        csr_data[0] = 0;
        for (std::size_t j = 1; j < layer_len; ++j) {
            csr_data[j] = csr_data[j-1] + child_counts[j-1];
        }

        long int* indices_data = new long int[layer_size];
        for (const auto *node: c.layers[i]) {
            std::size_t offset = 0;
            for (Node *child: node->children) {
                assert(child->layer == i-1);
                indices_data[csr_data[node->ix] + offset++] = child->ix;
            }
        }

        std::size_t indices_size[1] = {layer_size};
        std::size_t csr_size[1] = {layer_len};
        nb::capsule indices_capsule(indices_data, cleanup);
        nb::capsule csr_capsule(csr_data, cleanup);

        nb::ndarray<nb::numpy, long int, nb::shape<-1>> indices_ndarray(indices_data, 1, indices_size, indices_capsule);
        nb::ndarray<nb::numpy, long int, nb::shape<-1>> csr_ndarray(csr_data, 1, csr_size, csr_capsule);
        indices_ndarrays.push_back(indices_ndarray);
        csr_ndarrays.push_back(csr_ndarray);
    }

    return std::make_pair(indices_ndarrays, csr_ndarrays);
}

}  // namespace klay
