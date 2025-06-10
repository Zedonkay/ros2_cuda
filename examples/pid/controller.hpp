#pragma once

// Constants
constexpr int NUM_JOINTS = 6;
constexpr float DT = 0.01f;
constexpr float MAX_INTEGRAL = 100.0f;
constexpr float MIN_INTEGRAL = -100.0f;

// Data Structures
struct JointState {
    float position;
    float velocity;
    float integral;
};
struct PIDGains {
    float kp;
    float ki;
    float kd;
};
struct Control {
    float effort;
};

class PIDControllerNode {
public:
    // ROS-compatible method declarations
    virtual return fmaxf() = 0;
    virtual thrust::device_vector<JointState> d_states() = 0;
    virtual thrust::device_vector<PIDGains> d_gains() = 0;
    virtual thrust::device_vector<float> d_current_positions() = 0;
    virtual thrust::device_vector<float> d_current_velocities() = 0;
    virtual thrust::device_vector<float> d_target_positions() = 0;
    virtual thrust::device_vector<float> d_target_velocities() = 0;
    virtual thrust::device_vector<Control> d_controls() = 0;
    // Device functions as abstract interface
    // Global (kernel) functions as abstract interface
}; 