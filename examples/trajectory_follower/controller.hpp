#pragma once

// Constants
constexpr int STATE_DIM = 12;
constexpr int CONTROL_DIM = 6;
constexpr float DT = 0.01f;
constexpr float LOOKAHEAD_TIME = 0.5f;
constexpr float Kp = 1.0f;
constexpr float Kd = 0.1f;
constexpr float Ko = 1.0f;
constexpr float Kw = 0.1f;

// Data Structures
struct Trajectory {
    int num_states;
    float time_step;
};

class TrajectoryFollowerNode {
public:
    // ROS-compatible method declarations
    virtual return sqrtf() = 0;
    virtual thrust::device_vector<State> d_current_states() = 0;
    virtual thrust::device_vector<Trajectory> d_trajectories() = 0;
    virtual thrust::device_vector<Control> d_prev_controls() = 0;
    virtual thrust::device_vector<Control> d_controls() = 0;
    // Device functions as abstract interface
    // Global (kernel) functions as abstract interface
}; 