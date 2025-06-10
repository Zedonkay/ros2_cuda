#pragma once

// Constants
constexpr int STATE_DIM = 12;
constexpr int CONTROL_DIM = 6;
constexpr int NUM_CONTROLLERS = 1;


class LQRControllerNode {
public:
    // ROS-compatible method declarations
    virtual thrust::device_vector<State> d_current_states() = 0;
    virtual thrust::device_vector<State> d_target_states() = 0;
    virtual thrust::device_vector<LQRGains> d_gains() = 0;
    virtual thrust::device_vector<Control> d_prev_controls() = 0;
    virtual thrust::device_vector<Control> d_controls() = 0;
    // Device functions as abstract interface
    // Global (kernel) functions as abstract interface
}; 