#include "stretch_reference.h"

#ifndef __CUDACC__
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <algorithm>
#include <cassert>
#include <cmath>

namespace {

using Vec3f = Eigen::Vector3f;
using Mat3x2f = Eigen::Matrix<float, 3, 2>;
using Mat2x2f = Eigen::Matrix2f;
using Mat6x9f = Eigen::Matrix<float, 6, 9>;
using Vec6f = Eigen::Matrix<float, 6, 1>;
using VecXf = Eigen::VectorXf;
using SparseMat = Eigen::SparseMatrix<float>;
using Triplet = Eigen::Triplet<float>;

static Vec3f get_vertex(const VecXf& x, int idx)
{
    return Vec3f(x(idx * 3 + 0), x(idx * 3 + 1), x(idx * 3 + 2));
}

static Mat3x2f deformation_gradient(const VecXf& x,
                                    int i0, int i1, int i2,
                                    const Mat2x2f& inv_duv)
{
    Mat3x2f p;
    p.col(0) = get_vertex(x, i1) - get_vertex(x, i0);
    p.col(1) = get_vertex(x, i2) - get_vertex(x, i0);
    return p * inv_duv;
}

// DiffCloth Triangle::projectToManifold
static Mat3x2f project_to_manifold(const Mat3x2f& F)
{
    Mat3x2f basis;
    basis.setZero();

    Vec3f c0 = F.col(0);
    if (c0.norm() > 1e-10f) basis.col(0) = c0.normalized();
    else                    basis.col(0) = Vec3f(1.0f, 0.0f, 0.0f);

    Vec3f c1 = F.col(1) - F.col(1).dot(basis.col(0)) * basis.col(0);
    if (c1.norm() > 1e-10f) basis.col(1) = c1.normalized();
    else {
        Vec3f fallback(-basis(1,0), basis(0,0), 0.0f);
        if (fallback.norm() < 1e-10f)
            fallback = Vec3f(0.0f, -basis(2,0), basis(1,0));
        basis.col(1) = fallback.normalized();
    }

    Mat2x2f F2d = basis.transpose() * F;
    Eigen::JacobiSVD<Mat2x2f> svd(F2d, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Mat2x2f R2d = svd.matrixU() * svd.matrixV().transpose();

    // Enforce proper rotation if needed.
    if (R2d.determinant() < 0.0f) {
        Mat2x2f U = svd.matrixU();
        U.col(1) *= -1.0f;
        R2d = U * svd.matrixV().transpose();
    }

    return basis * R2d;
}

static Vec6f flatten_3x2(const Mat3x2f& M)
{
    Vec6f out;
    out.segment<3>(0) = M.col(0);
    out.segment<3>(3) = M.col(1);
    return out;
}

} // namespace

CpuStretchReferenceSolver::CpuStretchReferenceSolver(const ClothMesh& mesh,
                                                     const SimConstraints& sim_cons,
                                                     const Constraints& /*pin_cons*/,
                                                     float dt,
                                                     float gravity,
                                                     float damping)
    : num_verts_(mesh.num_verts)
    , num_tris_(mesh.num_tris)
    , dt_(dt)
    , gravity_(gravity)
    , damping_(damping)
{
    masses_ = mesh.mass;
    rest_area_ = mesh.rest_area;
    stretch_k_ = sim_cons.tri_stretch_k;

    tris_.resize(num_tris_ * 3);
    inv_duv_.resize(num_tris_ * 4);
    dFdx_.resize(num_tris_ * 54);

    for (int t = 0; t < num_tris_; ++t) {
        tris_[t * 3 + 0] = mesh.triangles[t](0);
        tris_[t * 3 + 1] = mesh.triangles[t](1);
        tris_[t * 3 + 2] = mesh.triangles[t](2);

        inv_duv_[t * 4 + 0] = mesh.Dm_inv[t](0, 0);
        inv_duv_[t * 4 + 1] = mesh.Dm_inv[t](1, 0);
        inv_duv_[t * 4 + 2] = mesh.Dm_inv[t](0, 1);
        inv_duv_[t * 4 + 3] = mesh.Dm_inv[t](1, 1);

        const Mat6x9f& deriv = mesh.dF_dx[t];
        for (int r = 0; r < 6; ++r)
            for (int c = 0; c < 9; ++c)
                dFdx_[t * 54 + r * 9 + c] = deriv(r, c);
    }
}

void CpuStretchReferenceSolver::project_triangle_to_manifold(const float* x, int t, float* out_proj6) const
{
    VecXf local_x(9);
    const int i0 = tris_[t * 3 + 0];
    const int i1 = tris_[t * 3 + 1];
    const int i2 = tris_[t * 3 + 2];
    for (int k = 0; k < 3; ++k) {
        local_x(k + 0) = x[i0 * 3 + k];
        local_x(k + 3) = x[i1 * 3 + k];
        local_x(k + 6) = x[i2 * 3 + k];
    }

    Mat2x2f inv;
    inv(0,0) = inv_duv_[t * 4 + 0];
    inv(1,0) = inv_duv_[t * 4 + 1];
    inv(0,1) = inv_duv_[t * 4 + 2];
    inv(1,1) = inv_duv_[t * 4 + 3];

    Mat3x2f F = deformation_gradient(local_x, 0, 1, 2, inv);
    Vec6f proj = flatten_3x2(project_to_manifold(F));
    for (int i = 0; i < 6; ++i) out_proj6[i] = proj(i);
}

void CpuStretchReferenceSolver::step(ClothMesh& mesh, const Constraints& pin_cons)
{
    const int dof = num_verts_ * 3;
    const float h2 = dt_ * dt_;

    VecXf x_old(dof), y(dof), rhs(dof);
    for (int i = 0; i < num_verts_; ++i) {
        const Vec3f pos = mesh.pos_cpu[i];
        const Vec3f vel = mesh.vel_cpu[i];
        x_old.segment<3>(i * 3) = pos;
        y(i * 3 + 0) = pos.x() + dt_ * vel.x();
        y(i * 3 + 1) = pos.y() + dt_ * vel.y() + h2 * gravity_;
        y(i * 3 + 2) = pos.z() + dt_ * vel.z();
    }

    std::vector<float> proj_storage(num_tris_ * 6, 0.0f);
    for (int iter = 0; iter < 1; ++iter) {
        for (int t = 0; t < num_tris_; ++t)
            project_triangle_to_manifold(x_old.data(), t, &proj_storage[t * 6]);

        std::vector<Triplet> trips;
        trips.reserve(num_verts_ * 3 + num_tris_ * 81);
        rhs.setZero();

        for (int i = 0; i < num_verts_; ++i) {
            const float m = masses_[i];
            for (int d = 0; d < 3; ++d) {
                const int idx = i * 3 + d;
                trips.emplace_back(idx, idx, m);
                rhs(idx) += m * y(idx);
            }
        }

        for (int t = 0; t < num_tris_; ++t) {
            const float weight = std::sqrt(std::max(0.0f, stretch_k_[t] * rest_area_[t]));
            if (weight == 0.0f) continue;

            Mat6x9f A;
            for (int r = 0; r < 6; ++r)
                for (int c = 0; c < 9; ++c)
                    A(r, c) = weight * dFdx_[t * 54 + r * 9 + c];

            Vec6f p;
            for (int i = 0; i < 6; ++i)
                p(i) = weight * proj_storage[t * 6 + i];

            Eigen::Matrix<float, 9, 9> AtA = A.transpose() * A;
            Eigen::Matrix<float, 9, 1> Atp = A.transpose() * p;

            int ids[3] = {tris_[t * 3 + 0], tris_[t * 3 + 1], tris_[t * 3 + 2]};
            for (int lv = 0; lv < 3; ++lv) {
                for (int ld = 0; ld < 3; ++ld) {
                    const int gi = ids[lv] * 3 + ld;
                    const int li = lv * 3 + ld;
                    rhs(gi) += h2 * Atp(li);
                    for (int rv = 0; rv < 3; ++rv) {
                        for (int rd = 0; rd < 3; ++rd) {
                            const int gj = ids[rv] * 3 + rd;
                            const int lj = rv * 3 + rd;
                            trips.emplace_back(gi, gj, h2 * AtA(li, lj));
                        }
                    }
                }
            }
        }

        SparseMat K(dof, dof);
        K.setFromTriplets(trips.begin(), trips.end(), [](float a, float b) { return a + b; });
        K.makeCompressed();

        // Strongly enforce pinned vertices by zeroing row/col and setting diag=1.
        for (int idx : pin_cons.pinned_indices) {
            assert(idx >= 0 && idx < num_verts_);
            for (int d = 0; d < 3; ++d) {
                const int row = idx * 3 + d;
                for (int col = 0; col < K.outerSize(); ++col) {
                    for (SparseMat::InnerIterator it(K, col); it; ++it) {
                        if (it.row() == row || it.col() == row) {
                            if (it.row() == row && it.col() == row) it.valueRef() = 1.0f;
                            else it.valueRef() = 0.0f;
                        }
                    }
                }
                rhs(row) = pin_cons.target_positions[row];
            }
        }

        Eigen::SimplicialLDLT<SparseMat> solver;
        solver.compute(K);
        if (solver.info() != Eigen::Success) return;
        VecXf x_new = solver.solve(rhs);
        if (solver.info() != Eigen::Success) return;

        for (int i = 0; i < num_verts_; ++i) {
            Vec3f oldp = get_vertex(x_old, i);
            Vec3f newp = get_vertex(x_new, i);
            mesh.vel_cpu[i] = (newp - oldp) * ((1.0f - damping_) / dt_);
            mesh.pos_cpu[i] = newp;
            x_old.segment<3>(i * 3) = newp;
        }
    }
}

#endif
