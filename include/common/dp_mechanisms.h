#ifndef DP_MECHANISMS_H
#define DP_MECHANISMS_H

#include <vector>
#include <random>
#include <cmath>
#include <stdexcept>

// Laplace mechanism for differential privacy
inline std::vector<double> add_laplace_noise(const std::vector<uint32_t>& aggregate,
                                              double epsilon,
                                              uint32_t sensitivity) {
    std::random_device rd;
    std::mt19937_64 gen(rd());

    double scale = static_cast<double>(sensitivity) / epsilon;
    std::exponential_distribution<double> exp_dist(1.0 / scale);
    std::uniform_real_distribution<double> sign_dist(0.0, 1.0);

    std::vector<double> noisy_aggregate(aggregate.size());

    for (size_t i = 0; i < aggregate.size(); i++) {
        double noise = exp_dist(gen);
        if (sign_dist(gen) < 0.5) {
            noise = -noise;
        }
        noisy_aggregate[i] = static_cast<double>(aggregate[i]) + noise;
    }

    return noisy_aggregate;
}

// Gaussian mechanism for (epsilon, delta)-differential privacy
inline std::vector<double> add_gaussian_noise(const std::vector<uint32_t>& aggregate,
                                               double epsilon,
                                               double delta,
                                               uint32_t sensitivity) {
    if (delta <= 0 || delta >= 1) {
        throw std::runtime_error("Delta must be in (0, 1)");
    }

    double scale = sensitivity * std::sqrt(2.0 * std::log(1.25 / delta)) / epsilon;

    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::normal_distribution<double> gauss(0.0, scale);

    std::vector<double> noisy_aggregate(aggregate.size());
    for (size_t i = 0; i < aggregate.size(); i++) {
        noisy_aggregate[i] = static_cast<double>(aggregate[i]) + gauss(gen);
    }

    return noisy_aggregate;
}

#endif // DP_MECHANISMS_H
