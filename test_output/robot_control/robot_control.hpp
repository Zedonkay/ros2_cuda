#pragma once

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <vector>
#include <cmath>
#include <random>

// Constants
constexpr int DOF = 6;
constexpr float PI = 3.14159265358979323846f;
constexpr int MPPI_SAMPLES = 1000;
constexpr int MPC_HORIZON = 10;
constexpr int RRT_MAX_ITERATIONS = 1000;

// Data Structures
struct JointAngles {
    float angles[DOF];
    float velocities[DOF];
    float accelerations[DOF];
};
struct Pose {
    float position[3];
    float orientation[4];
    float velocity[6];
};
struct ControlGains {
    float kp[DOF];
    float ki[DOF];
    float kd[DOF];
};

class RobotControlNode {
public:
    virtual ~RobotControlNode() = default;
    
    // ROS-compatible method declarations
    
    // Device functions as abstract interface
    virtual float evaluate_sample_cost(const JointAngles& sample, const JointAngles& current_state, const Pose& target_pose) = 0;
    
    // Global (kernel) functions as abstract interface
}; 