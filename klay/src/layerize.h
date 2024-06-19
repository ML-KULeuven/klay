#ifndef MYPYBINDMODULE_HELLO_H
#define MYPYBINDMODULE_HELLO_H

#include <pybind11/pybind11.h>
namespace py = pybind11;


#include <string>

void brr(const std::string &filename);

struct Node;

Node* createAndNode();
Node* createOrNode();

#endif
