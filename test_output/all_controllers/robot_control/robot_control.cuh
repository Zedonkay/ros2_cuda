#pragma once

#include <cuda_runtime.h>
#include "RobotControlNode.hpp"



// Device function declarations
__device__ float evaluate_sample_cost(const JointAngles& sample, const JointAngles& current_state, const Pose& target_pose);


class RobotControlNode_Impl : public RobotControlNode {
public:
    RobotControlNode_Impl() = default;
    ~RobotControlNode_Impl() override = default;


    // Device function overrides
    float evaluate_sample_cost(const JointAngles& sample, const JointAngles& current_state, const Pose& target_pose) override;

}; 