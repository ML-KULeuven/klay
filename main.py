from graphviz import Source

import klay


def main():
    klay.brr(name="test.sdd")

    s = Source.from_file("tensorized.dot")
    s.view()
    s = Source.from_file("layerized.dot")
    s.view()


if __name__ == "__main__":
    main()
