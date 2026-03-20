#pragma once

enum class MaterialModel { StVK, NeoHookean };

struct MaterialParams {
    MaterialModel model         = MaterialModel::StVK;
    float young_modulus         = 1e5f;   // Pa
    float poisson_ratio         = 0.3f;
    float density               = 0.1f;   // kg/m²
    float damping_alpha         = 0.01f;  // Rayleigh mass coefficient
    float damping_beta          = 0.001f; // Rayleigh stiffness coefficient

    float lambda() const {
        float E = young_modulus, nu = poisson_ratio;
        return E * nu / ((1.0f + nu) * (1.0f - 2.0f * nu));
    }

    float mu() const {
        float E = young_modulus, nu = poisson_ratio;
        return E / (2.0f * (1.0f + nu));
    }
};

struct BendingParams {
    float stiffness = 1e-3f;
};
