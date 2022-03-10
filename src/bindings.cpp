#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/pytypes.h>
//#include "smme.hpp" #gives import error in python: flat namespace...
#include "smme.cpp"

void bind_pga(py::module &m) {
    m.def("pga",
        &pga,
        R"pbdoc(
            proximal gradient algorithm for soft maximin estimation

            Parameters
            ----------
            arr : ...
                
            Returns
            -------
            coeffs: ...
        )pbdoc"
    );
     m.def("WT",
        &WT,
        R"pbdoc(
            wavelet transform

            Parameters
            ----------
            arr :...

            Returns
            -------
            coeffs: ...
        )pbdoc"
    );
     m.def(
        "IWT",
        &IWT,
        R"pbdoc(
            inverse wavelet transform

            Parameters
            ----------
            np.ndarray
                input array

            Returns
            -------
            np.ndarray
                output array
        )pbdoc"
    );
}
PYBIND11_MODULE(_smme, m) {
    bind_pga(m);
}
