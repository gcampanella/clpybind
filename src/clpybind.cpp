#include <coin/ClpSimplex.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

typedef py::array_t<double, py::array::c_style | py::array::forcecast> np_double_array;

class DenseMatrix : public CoinPackedMatrix {
public:
    DenseMatrix(np_double_array x) {
        py::buffer_info info = x.request();
        size_t nRows = static_cast<size_t>(info.shape[0]);
        size_t nCols = static_cast<size_t>(info.shape[1]);
        size_t nElements = static_cast<size_t>(info.shape[0] * info.shape[1]);

        colOrdered_ = false;
        extraGap_ = 0;
        extraMajor_ = 0;
        element_ = reinterpret_cast<double *>(info.ptr);
        index_ = new int[nElements];
        start_ = new CoinBigIndex[nRows];
        length_ = new int[nRows];
        majorDim_ = static_cast<int>(nRows);
        minorDim_ = static_cast<int>(nCols);
        size_ = static_cast<CoinBigIndex>(nElements);
        maxMajorDim_ = majorDim_;
        maxSize_ = size_;

        size_t i, j;
        for (i = 0; i < nRows; ++i) {
            start_[i] = static_cast<int>(i * nCols);
            length_[i] = static_cast<int>(nCols);
            for (j = 0; j < nCols; ++j) {
                index_[i*nCols + j] = static_cast<int>(j);
            }
        }
    }

    ~DenseMatrix() override {
        element_ = nullptr; // Avoid GC in ~CoinPackedMatrix()
    }

    py::buffer_info get_buffer_info() {
        return py::buffer_info(element_, sizeof(double), py::format_descriptor<double>::format(),
                               2, { majorDim_, minorDim_ }, { sizeof(double) * minorDim_, sizeof(double) } );
    }
};

PYBIND11_MODULE(clpybind, m) {
    py::class_<DenseMatrix> matrix(m, "Matrix", py::buffer_protocol());
    matrix.def(py::init<np_double_array>(), py::keep_alive<1, 2>());
    matrix.def_buffer(&DenseMatrix::get_buffer_info);

    py::class_<ClpSimplex> simplex(m, "Simplex", py::buffer_protocol());

    simplex.def(py::init([](const DenseMatrix &matrix,
                            np_double_array colLower, np_double_array colUpper,
                            np_double_array obj,
                            np_double_array rowLower, np_double_array rowUpper) {
        py::buffer_info colLowerInfo = colLower.request();
        if (colLowerInfo.ndim != 1 || colLowerInfo.shape[0] != matrix.getMinorDim()) {
            throw std::invalid_argument("Invalid column lower bounds");
        }
        py::buffer_info colUpperInfo = colUpper.request();
        if (colLowerInfo.ndim != 1 || colLowerInfo.shape[0] != matrix.getMinorDim()) {
            throw std::invalid_argument("Invalid column upper bounds");
        }
        py::buffer_info objInfo = obj.request();
        if (objInfo.ndim != 1 || objInfo.shape[0] != matrix.getMinorDim()) {
            throw std::invalid_argument("Invalid objective");
        }
        py::buffer_info rowLowerInfo = rowLower.request();
        if (rowLowerInfo.ndim != 1 || rowLowerInfo.shape[0] != matrix.getMajorDim()) {
            throw std::invalid_argument("Invalid row lower bounds");
        }
        py::buffer_info rowUpperInfo = rowUpper.request();
        if (rowUpperInfo.ndim != 1 || rowUpperInfo.shape[0] != matrix.getMajorDim()) {
            throw std::invalid_argument("Invalid row upper bounds");
        }
        ClpSimplex *simplex = new ClpSimplex;
        simplex->loadProblem(matrix,
                             reinterpret_cast<const double *>(colLowerInfo.ptr),
                             reinterpret_cast<const double *>(colUpperInfo.ptr),
                             reinterpret_cast<const double *>(objInfo.ptr),
                             reinterpret_cast<const double *>(rowLowerInfo.ptr),
                             reinterpret_cast<const double *>(rowUpperInfo.ptr));
        return simplex;
    }));

    enum class LogLevel {
        Off = 0,
        FinalOnly = 1,
        FactorizationsOnly = 2,
        FactorizationsAndABitMore = 3,
        Verbose = 4
    };

    py::enum_<LogLevel>(simplex, "LogLevel")
        .value("Off", LogLevel::Off)
        .value("FinalOnly", LogLevel::FinalOnly)
        .value("FactorizationsOnly", LogLevel::FactorizationsOnly)
        .value("FactorizationsAndABitMore", LogLevel::FactorizationsAndABitMore)
        .value("Verbose", LogLevel::Verbose)
        .export_values();

    simplex.def_property("log_level",
                         &ClpSimplex::logLevel, &ClpSimplex::setLogLevel);

    simplex.def_property("max_iterations",
                         &ClpSimplex::maximumIterations, &ClpSimplex::setMaximumIterations);

    simplex.def_property("max_seconds",
                         &ClpSimplex::maximumSeconds, &ClpSimplex::setMaximumSeconds);

    enum class Scaling {
        Off = 0,
        Equilibrium = 1,
        Geometric = 2,
        Automatic = 3
    };

    py::enum_<Scaling>(simplex, "Scaling")
        .value("Off", Scaling::Off)
        .value("Equilibrium", Scaling::Equilibrium)
        .value("Geometric", Scaling::Geometric)
        .value("Automatic", Scaling::Automatic)
        .export_values();

    simplex.def_property("scaling",
                         [](ClpSimplex &simplex) -> Scaling { return Scaling(simplex.scalingFlag()); },
                         [](ClpSimplex &simplex, Scaling scaling) { simplex.scaling(static_cast<int>(scaling)); });

    enum class OptimizationDirection {
        Minimize = 1,
        Maximize = -1,
        Ignore = 0
    };

    py::enum_<OptimizationDirection>(simplex, "OptimizationDirection")
        .value("Minimize", OptimizationDirection::Minimize)
        .value("Maximize", OptimizationDirection::Maximize)
        .value("Ignore", OptimizationDirection::Ignore)
        .export_values();

    simplex.def_property("optimization_direction",
                         [](ClpSimplex &simplex) -> OptimizationDirection {
                             return OptimizationDirection(static_cast<int>(simplex.optimizationDirection()));
                         },
                         [](ClpSimplex &simplex, OptimizationDirection direction) {
                             simplex.setOptimizationDirection(static_cast<double>(direction));
                         });

    simplex.def_property("dual_tolerance",
                         &ClpSimplex::dualTolerance, &ClpSimplex::setDualTolerance);

    simplex.def_property("primal_tolerance",
                         &ClpSimplex::primalTolerance, &ClpSimplex::setPrimalTolerance);

    enum class ProblemStatus {
        Unknown = -1,
        Optimal = 0,
        PrimalInfeasible = 1,
        DualInfeasible = 2,
        StoppedOnTimeout = 3,
        StoppedOnError = 4,
        StoppedOnEventHandler = 5
    };

    py::enum_<ProblemStatus>(simplex, "ProblemStatus")
            .value("Unknown", ProblemStatus::Unknown)
            .value("Optimal", ProblemStatus::Optimal)
            .value("PrimalInfeasible", ProblemStatus::PrimalInfeasible)
            .value("DualInfeasible", ProblemStatus::DualInfeasible)
            .value("StoppedOnTimeout", ProblemStatus::StoppedOnTimeout)
            .value("StoppedOnError", ProblemStatus::StoppedOnError)
            .value("StoppedOnEventHandler", ProblemStatus::StoppedOnEventHandler)
            .export_values();

    simplex.def("initial_solve", [](ClpSimplex &simplex, bool dual) -> ProblemStatus {
        int status = dual ? simplex.initialDualSolve() :
                            simplex.initialPrimalSolve();
        return ProblemStatus(status);
    }, py::arg("dual") = true);

    simplex.def_property_readonly("status", [](ClpSimplex &simplex) -> ProblemStatus {
        return ProblemStatus(simplex.problemStatus());
    }, py::return_value_policy::reference_internal);

    simplex.def_property_readonly("objective_value",
                                  &ClpSimplex::objectiveValue,
                                  py::return_value_policy::reference_internal);

    simplex.def_property_readonly("solution", [](ClpSimplex &simplex) -> np_double_array {
        return np_double_array(static_cast<size_t>(simplex.numberColumns()), simplex.primalColumnSolution());
    }, py::return_value_policy::reference_internal);

    simplex.def_property_readonly("reduced_costs", [](ClpSimplex &simplex) -> np_double_array {
        return np_double_array(static_cast<size_t>(simplex.numberColumns()), simplex.dualColumnSolution());
    }, py::return_value_policy::reference_internal);

    simplex.def_property_readonly("row_activities", [](ClpSimplex &simplex) -> np_double_array {
        return np_double_array(static_cast<size_t>(simplex.numberRows()), simplex.primalRowSolution());
    }, py::return_value_policy::reference_internal);

    simplex.def_property_readonly("shadow_prices", [](ClpSimplex &simplex) -> np_double_array {
        return np_double_array(static_cast<size_t>(simplex.numberRows()), simplex.dualRowSolution());
    }, py::return_value_policy::reference_internal);
}
