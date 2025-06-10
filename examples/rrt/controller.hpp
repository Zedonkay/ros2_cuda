#pragma once

// Constants
constexpr int STATE_DIM = 12;
constexpr int MAX_NODES = 1000;
constexpr float GOAL_BIAS = 0.1f;
constexpr float STEP_SIZE = 0.1f;
constexpr float GOAL_THRESHOLD = 0.1f;

// Data Structures
struct Node {
    State state;
    int parent;
    float cost;
};
struct Trajectory {
    int num_states;
};

class RRTPlannerNode {
public:
    // ROS-compatible method declarations
    virtual return sqrtf() = 0;
    virtual thrust::uniform_real_distribution<float> pos_dist() = 0;
    virtual thrust::uniform_real_distribution<float> vel_dist() = 0;
    virtual thrust::uniform_real_distribution<float> ori_dist() = 0;
    virtual thrust::uniform_real_distribution<float> ang_vel_dist() = 0;
    virtual thrust::uniform_real_distribution<float> bias_dist() = 0;
    virtual return compute_distance() = 0;
    virtual thrust::device_vector<Node> d_nodes() = 0;
    virtual thrust::device_vector<int> d_num_nodes() = 0;
    virtual thrust::device_vector<int> d_goal_idx() = 0;
    virtual thrust::device_vector<Trajectory> d_trajectory() = 0;
    // Device functions as abstract interface
    // Global (kernel) functions as abstract interface
}; 