#pragma once

// Constants
constexpr int DOF = 6;
constexpr float PI = 3.14159265358979323846f;
constexpr int MPPI_SAMPLES = 1000;
constexpr int MPC_HORIZON = 10;
constexpr int RRT_MAX_ITERATIONS = 1000;


class RobotControlNode {
public:
    // ROS-compatible method declarations
    virtual thrust::normal_distribution<float> dist() = 0;
    virtual thrust::device_vector<float> costs() = 0;
    virtual thrust::device_vector<JointAngles> samples() = 0;
    virtual thrust::device_vector<float> error(DOF * 2) = 0;
    virtual thrust::device_vector<float> states(horizon * DOF * 2) = 0;
    virtual thrust::device_vector<float> controls(horizon * DOF) = 0;
    virtual return sqrtf() = 0;
    virtual return sqrtf() = 0;
    virtual PIDController pid() = 0;
    virtual TrajectoryFollower follower() = 0;
    // Device functions as abstract interface
    // Global (kernel) functions as abstract interface
}; 