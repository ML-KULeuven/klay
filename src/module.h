#ifndef NANOBIND_EXAMPLE_MODULE_H
#define NANOBIND_EXAMPLE_MODULE_H

struct Node;
struct Circuit;

Node* createOrNode();
Node* createAndNode();
Node* createTrueNode();
Node* createFalseNode();

void to_dot_file(Circuit& circuit, const std::string& filename);

typedef std::vector<nb::ndarray<nb::numpy, long int, nb::shape<-1>>> Arrays;

Node* parseSDDFile(const std::string& filename, Circuit& circuit);
void layerize(std::vector<Node*> nodes, Circuit& circuit);
std::pair<Arrays, Arrays> tensorize(Circuit& circuit);

#endif //NANOBIND_EXAMPLE_MODULE_H
