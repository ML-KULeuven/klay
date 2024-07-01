#ifndef NANOBIND_EXAMPLE_MODULE_H
#define NANOBIND_EXAMPLE_MODULE_H

struct Node;
struct Circuit;

Node* createOrNode();
Node* createAndNode();

typedef std::vector<std::vector<long int>> Arrays;

void parseSDDFile(const std::string& filename, Circuit& circuit);
void layerize(std::vector<Node*> nodes, Circuit& circuit);
std::pair<Arrays, Arrays> tensorize(Circuit& circuit);

#endif //NANOBIND_EXAMPLE_MODULE_H
