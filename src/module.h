#ifndef NANOBIND_EXAMPLE_MODULE_H
#define NANOBIND_EXAMPLE_MODULE_H

struct Node;
struct Circuit;

Node* createOrNode();
Node* createAndNode();

unsigned int parseSDDFile(const std::string& filename, std::vector<Node*>& nodes);
void layerize(std::vector<Node*> nodes, Circuit& circuit);
void tensorize(Circuit& circuit);

#endif //NANOBIND_EXAMPLE_MODULE_H
