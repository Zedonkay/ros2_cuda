#pragma once

// Constants
constexpr int STATE_DIM = 12;
constexpr int CONTROL_DIM = 6;
constexpr int HORIZON = 50;
constexpr float DT = 0.01f;
constexpr int MAX_ITERATIONS = 100;
constexpr float LEARNING_RATE = 0.01f;

// Data Structures
struct MPCParams {
    float max_force;
    float max_torque;
    float max_rate;
};

class MPCControllerNode {
public:
    // ROS-compatible method declarations
    virtual thrust::device_vector<Control> d_controls() = 0;
    // Device functions as abstract interface
    // Global (kernel) functions as abstract interface
}; 