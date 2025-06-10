#pragma once

// Constants
constexpr int STATE_DIM = 12;
constexpr int CONTROL_DIM = 6;
constexpr int NUM_SAMPLES = 1000;
constexpr float DT = 0.01f;
constexpr int HORIZON = 50;
constexpr float NOISE_SCALE = 0.1f;
constexpr float TEMPERATURE = 1.0f;

// Data Structures
struct Trajectory {
    float cost;
};

class MPPIControllerNode {
public:
    // ROS-compatible method declarations
    virtual thrust::normal_distribution<float> dist() = 0;
    virtual return expf() = 0;
    virtual thrust::device_vector<Trajectory> d_trajectories() = 0;
    virtual thrust::device_vector<Control> d_nominal_controls() = 0;
    virtual thrust::device_vector<float> d_costs() = 0;
    // Device functions as abstract interface
    // Global (kernel) functions as abstract interface
}; 