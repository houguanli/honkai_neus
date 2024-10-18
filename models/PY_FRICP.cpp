#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "FRICP.h"
#include "ICP.h"
#include "io_pc.h"

namespace py = pybind11;

template <typename T>
class PY_FRICP {
public:
    typedef Eigen::Matrix<T, 3, Eigen::Dynamic> Vertices;
    typedef Eigen::Matrix<T, 3, 1> VectorN;
    PY_FRICP();
    ~PY_FRICP() {}
    void set_source_points(py::array_t<T> &source_point);
    void set_target_points(py::array_t<T> &target_point);
    void set_source_from_file(std::string file_source);
    void set_target_from_file(std::string file_target);
    void set_points(py::array_t<T> &source_point, py::array_t<T> &target_point);
    void set_points_from_file(std::string file_source, std::string file_target);
    void numpy_to_eigen(py::array_t<T> &arr);
    void set_init_matrix(py::array_t<T> &init_matrix);
public:
    py::array_t<T> run_icp(unsigned int method = 3);   // TODO
private:
    Vertices vertices;
    Vertices vertices_source, normal_source, src_vert_colors;
    Vertices vertices_target, normal_target, tar_vert_colors;
    MatrixXX init_matrix;
    MatrixXX res_trans;
    enum Method{ICP, AA_ICP, FICP, RICP, PPL, RPPL, SparseICP, SICPPPL} m_method=RICP;
    bool use_init{false};
};

template <typename T>
PY_FRICP<T>::PY_FRICP() {
    //set initial transformation to identity matrix
    init_matrix = MatrixXX::Identity(4, 4);
}

template <typename T>
void PY_FRICP<T>::set_source_points(py::array_t<T> &source_point) {
    numpy_to_eigen(source_point);
    this->vertices_source = this->vertices;
    std::cout << "source: " << this->vertices_source.rows() << "x" << this->vertices_source.cols() << std::endl;
}

template <typename T>
void PY_FRICP<T>::set_target_points(py::array_t<T> &target_point) {
    numpy_to_eigen(target_point);
    this->vertices_target = this->vertices;
    std::cout << "target: " << this->vertices_target.rows() << "x" << this->vertices_target.cols() << std::endl;
}

template <typename T>
void PY_FRICP<T>::set_source_from_file(std::string file_source) {
    if(file_source.size()==0)
    {
        throw std::runtime_error("Source file is empty");
        exit(0);
    }
    read_file(vertices_source, normal_source, src_vert_colors, file_source);
    std::cout << "source: " << vertices_source.rows() << "x" << vertices_source.cols() << std::endl;
}

template <typename T>
void PY_FRICP<T>::set_target_from_file(std::string file_target) {
    if(file_target.size()==0)
    {
        throw std::runtime_error("Target file is empty");
        exit(0);
    }
    read_file(vertices_target, normal_target, tar_vert_colors, file_target);
    std::cout << "target: " << vertices_target.rows() << "x" << vertices_target.cols() << std::endl;
}

template <typename T>
void PY_FRICP<T>::set_points_from_file(std::string file_source, std::string file_target) {
    set_source_from_file(file_source);
    set_target_from_file(file_target);
}

template <typename T>
void PY_FRICP<T>::set_points(py::array_t<T> &source_point, py::array_t<T> &target_point) {
    set_source_points(source_point);
    set_target_points(target_point);
}

template <typename T>
void PY_FRICP<T>::set_init_matrix(py::array_t<T> &init_matrix) {
    // assert that the array is in shape (4, 4)
    if (init_matrix.ndim() != 2) {
        throw std::runtime_error("Input should be a 2D array");
    }
    if (init_matrix.shape(0) != 4 || init_matrix.shape(1) != 4) {
        throw std::runtime_error("Input should have 4 rows and 4 columns");
    }
    use_init = true;
    // get the buffer info
    py::buffer_info info = init_matrix.request();
    T *ptr = static_cast<T *>(info.ptr);
    // copy the data to the Eigen matrix
    this->init_matrix = Eigen::Map<MatrixXX>(ptr, 4, 4);
    std::cout<< "set initial transformation matrix as: \n " << this->init_matrix << std::endl;
}

template <typename T>
void PY_FRICP<T>::numpy_to_eigen(py::array_t<T> &arr) {
    // assert that the array is in shape (3, N)
    if (arr.ndim() != 2) {
        throw std::runtime_error("Input should be a 2D array");
    }
    if (arr.shape(0) != 3) {
        throw std::runtime_error("Input should have 3 rows");
    }
    // get the buffer info
    py::buffer_info info = arr.request();
    T *ptr = static_cast<T *>(info.ptr);
    // copy the data to the Eigen matrix
    vertices = Eigen::Map<Vertices>(ptr, 3, info.shape[1]);
    // std::cout << "vertices: " << vertices.rows() << "x" << vertices.cols() << std::endl;
    // cout vertices data
    // std::cout << vertices << std::endl;
}

template <typename T>
py::array_t<T> PY_FRICP<T>::run_icp(unsigned int method)
{
    // assert that the source and target points are set
    if (vertices_source.size() == 0 || vertices_target.size() == 0) {
        throw std::runtime_error("Source and target points should be set first");
        return py::array_t<T>({4, 4});
    }    
    int dim = 3;

    // scaling
    Eigen::Vector3d source_scale, target_scale;
    source_scale = this->vertices_source.rowwise().maxCoeff() - this->vertices_source.rowwise().minCoeff();
    target_scale = this->vertices_target.rowwise().maxCoeff() - this->vertices_target.rowwise().minCoeff();
    double scale = std::max(source_scale.norm(), target_scale.norm());
    std::cout << "scale = " << scale << std::endl;
    this->vertices_source /= scale;
    this->vertices_target /= scale;

    /// De-mean
    VectorN source_mean, target_mean;
    source_mean = this->vertices_source.rowwise().sum() / double(vertices_source.cols());
    target_mean = vertices_target.rowwise().sum() / double(vertices_target.cols());
    this->vertices_source.colwise() -= source_mean;
    this->vertices_target.colwise() -= target_mean;

    double time;
    // set ICP parameters
    ICP::Parameters pars;

    // set Sparse-ICP parameters
    SICP::Parameters spars;
    spars.p = 0.4;
    spars.print_icpn = false;

    if(this->use_init)
    {
        this->init_matrix.block(0, dim, dim, 1) /= scale;
        this->init_matrix.block(0,3,3,1) += this->init_matrix.block(0,0,3,3)*source_mean - target_mean;
        pars.use_init = true;
        pars.init_trans = this->init_matrix;
        spars.init_trans = this->init_matrix;
    }

    std::cout << "begin registration..." << std::endl;
    FRICP<3> fricp;
    double begin_reg = omp_get_wtime();
    double converge_rmse = 0;
    switch(method)
    {
    case ICP:
    {
        pars.f = ICP::NONE;
        pars.use_AA = false;
        fricp.point_to_point(this->vertices_source, this->vertices_target, source_mean, target_mean, pars);
        this->res_trans = pars.res_trans;
        break;
    }
    case AA_ICP:
    {
        AAICP::point_to_point_aaicp(this->vertices_source, this->vertices_target, source_mean, target_mean, pars);
        this->res_trans = pars.res_trans;
        break;
    }
    case FICP:
    {
        pars.f = ICP::NONE;
        pars.use_AA = true;
        fricp.point_to_point(vertices_source, vertices_target, source_mean, target_mean, pars);
        this->res_trans = pars.res_trans;
        break;
    }
    case RICP:
    {
        pars.f = ICP::WELSCH;
        pars.use_AA = true;
        fricp.point_to_point(this->vertices_source, this->vertices_target, source_mean, target_mean, pars);
        this->res_trans = pars.res_trans;
        break;
    }
    case PPL:
    {
        pars.f = ICP::NONE;
        pars.use_AA = false;
        if(normal_target.size()==0)
        {
            std::cout << "Warning! The target model without normals can't run Point-to-plane method!" << std::endl;
            exit(0);
        }
        fricp.point_to_plane(this->vertices_source, this->vertices_target, normal_source, normal_target, source_mean, target_mean, pars);
        this->res_trans = pars.res_trans;
        break;
    }
    case RPPL:
    {
        pars.nu_end_k = 1.0/6;
        pars.f = ICP::WELSCH;
        pars.use_AA = true;
        if(normal_target.size()==0)
        {
            std::cout << "Warning! The target model without normals can't run Point-to-plane method!" << std::endl;
            exit(0);
        }
        fricp.point_to_plane_GN(this->vertices_source, this->vertices_target, normal_source, normal_target, source_mean, target_mean, pars);
        this->res_trans = pars.res_trans;
        break;
    }
    case SparseICP:
    {
        SICP::point_to_point(this->vertices_source, this->vertices_target, source_mean, target_mean, spars);
        this->res_trans = spars.res_trans;
        break;
    }
    case SICPPPL:
    {
        if(normal_target.size()==0)
        {
            std::cout << "Warning! The target model without normals can't run Point-to-plane method!" << std::endl;
            exit(0);
        }
        SICP::point_to_plane(this->vertices_source, this->vertices_target, normal_target, source_mean, target_mean, spars);
        this->res_trans = spars.res_trans;
        break;
    }
    }
    double end_reg = omp_get_wtime();
    time = end_reg - begin_reg;
    std::cout << "Registration done!" << std::endl;
    std::cout << "Time: " << time << "s" << std::endl;
    this->vertices_source = scale * vertices_source;
    this->vertices_target = scale * vertices_target;
    Eigen::Affine3d res_T;
    res_T.linear() = res_trans.block(0,0,3,3);
    res_T.translation() = res_trans.block(0,3,3,1);
    this->res_trans.block(0,3,3,1) *= scale;
    std::cout << "res_T: " << this->res_trans << std::endl;

    py::array_t<T> py_result = py::array_t<T>({4, 4});
    py::buffer_info info = py_result.request();
    T *ptr = static_cast<T *>(info.ptr);
    for(size_t i = 0; i < 4; i++)
        for(size_t j = 0; j < 4; j++)
            ptr[i * 4 + j] = this->res_trans(i, j);
    return py_result;
}

// instantiate the class for float and double
// template class PY_FRICP<float>;
template class PY_FRICP<double>;


PYBIND11_MODULE(py_fricp, m) {
    m.doc() = "Fast Robust ICP";
    // py::class_<PY_FRICP<float>>(m, "PY_FRICP_float")
    //     .def(py::init<>())
    //     .def("numpy_to_eigen", &PY_FRICP<float>::numpy_to_eigen, "Convert numpy array to Eigen matrix", py::arg("arr"))
    //     .def("set_source_points", &PY_FRICP<float>::set_source_points, "Set source points", py::arg("source_point"))
    //     .def("set_target_points", &PY_FRICP<float>::set_target_points, "Set target points", py::arg("target_point"))
    //     .def("set_points", &PY_FRICP<float>::set_points, "Set source and target points", py::arg("source_point"), py::arg("target_point"))
    //     .def("run_icp", &PY_FRICP<float>::run_icp, "Run ICP", py::arg("method") = 3)
    //     .def("set_init_matrix", &PY_FRICP<float>::set_init_matrix, "Set initial transformation matrix", py::arg("init_matrix"));
    py::class_<PY_FRICP<double>>(m, "PY_FRICPd")
        .def(py::init<>())
        .def("numpy_to_eigen", &PY_FRICP<double>::numpy_to_eigen, "Convert numpy array to Eigen matrix", py::arg("arr"))
        .def("set_source_points", &PY_FRICP<double>::set_source_points, "Set source points", py::arg("source_point"))
        .def("set_target_points", &PY_FRICP<double>::set_target_points, "Set target points", py::arg("target_point"))
        .def("set_points", &PY_FRICP<double>::set_points, "Set source and target points", py::arg("source_point"), py::arg("target_point"))
        .def("set_init_matrix", &PY_FRICP<double>::set_init_matrix, "Set initial transformation matrix", py::arg("init_matrix"))
        .def("run_icp", &PY_FRICP<double>::run_icp, "Run ICP", py::arg("method") = 3)
        .def("set_source_from_file", &PY_FRICP<double>::set_source_from_file, "Set source points from file", py::arg("file_source"))
        .def("set_target_from_file", &PY_FRICP<double>::set_target_from_file, "Set target points from file", py::arg("file_target"))
        .def("set_points_from_file", &PY_FRICP<double>::set_points_from_file, "Set source and target points from file", py::arg("file_source"), py::arg("file_target"));
}