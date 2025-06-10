// This file is auto-generated from a .cuif specification
// Class: TrajectoryFollowerNode
// Method: compute_control

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/iterator/zip_iterator.h>
#include <vector>
#include <cmath>

// Constants
constexpr int STATE_DIM = 12;  // 12-dimensional state vector
constexpr int CONTROL_DIM = 6;  // 6-dimensional control vector
constexpr float DT = 0.01f;  // Time step
constexpr float LOOKAHEAD_TIME = 0.5f;  // Lookahead time for trajectory following
constexpr float Kp = 1.0f;  // Position gain
constexpr float Kd = 0.1f;  // Velocity gain
constexpr float Ko = 1.0f;  // Orientation gain
constexpr float Kw = 0.1f;  // Angular velocity gain

// Data structures
struct State {
    float position[3];
    float velocity[3];
    float orientation[3];
    float angular_velocity[3];
};

struct Control {
    float force[3];
    float torque[3];
};

struct Trajectory {
    State states[1000];  // Maximum trajectory length
    int num_states;
    float time_step;
};

// Device functions for core computations
__device__ float compute_distance(const State& a, const State& b) {
    float dist = 0.0f;
    
    // Position distance
    for (int i = 0; i < 3; i++) {
        float pos_diff = a.position[i] - b.position[i];
        dist += pos_diff * pos_diff;
    }
    
    // Orientation distance
    for (int i = 0; i < 3; i++) {
        float ori_diff = a.orientation[i] - b.orientation[i];
        dist += ori_diff * ori_diff;
    }
    
    return sqrtf(dist);
}

__device__ int find_closest_trajectory_point(const Trajectory& trajectory, const State& current_state) {
    int closest_idx = 0;
    float min_dist = FLT_MAX;
    
    for (int i = 0; i < trajectory.num_states; i++) {
        float dist = compute_distance(trajectory.states[i], current_state);
        if (dist < min_dist) {
            min_dist = dist;
            closest_idx = i;
        }
    }
    
    return closest_idx;
}

__device__ void interpolate_state(const State& a, const State& b, float t, State& result) {
    for (int i = 0; i < 3; i++) {
        result.position[i] = a.position[i] + t * (b.position[i] - a.position[i]);
        result.velocity[i] = a.velocity[i] + t * (b.velocity[i] - a.velocity[i]);
        result.orientation[i] = a.orientation[i] + t * (b.orientation[i] - a.orientation[i]);
        result.angular_velocity[i] = a.angular_velocity[i] + t * (b.angular_velocity[i] - a.angular_velocity[i]);
    }
}

__device__ void compute_lookahead_state(const Trajectory& trajectory, const State& current_state, State& lookahead_state) {
    // Find closest point
    int closest_idx = find_closest_trajectory_point(trajectory, current_state);
    
    // Compute lookahead index
    float lookahead_steps = LOOKAHEAD_TIME / trajectory.time_step;
    int lookahead_idx = min(closest_idx + static_cast<int>(lookahead_steps), trajectory.num_states - 1);
    
    // Interpolate if needed
    if (lookahead_idx < trajectory.num_states - 1) {
        float t = lookahead_steps - static_cast<int>(lookahead_steps);
        interpolate_state(trajectory.states[lookahead_idx], 
                         trajectory.states[lookahead_idx + 1], 
                         t, 
                         lookahead_state);
    } else {
        lookahead_state = trajectory.states[lookahead_idx];
    }
}

__device__ void compute_control(const State& current_state, const State& target_state, Control& control) {
    // Position control
    for (int i = 0; i < 3; i++) {
        float pos_error = target_state.position[i] - current_state.position[i];
        float vel_error = target_state.velocity[i] - current_state.velocity[i];
        control.force[i] = Kp * pos_error + Kd * vel_error;
    }
    
    // Orientation control
    for (int i = 0; i < 3; i++) {
        float ori_error = target_state.orientation[i] - current_state.orientation[i];
        float ang_vel_error = target_state.angular_velocity[i] - current_state.angular_velocity[i];
        control.torque[i] = Ko * ori_error + Kw * ang_vel_error;
    }
}

__device__ void saturate_control(Control& control, float max_force, float max_torque) {
    for (int i = 0; i < 3; i++) {
        control.force[i] = fmaxf(fminf(control.force[i], max_force), -max_force);
        control.torque[i] = fmaxf(fminf(control.torque[i], max_torque), -max_torque);
    }
}

__device__ void limit_control_rate(Control& control, const Control& prev_control, float max_rate) {
    for (int i = 0; i < 3; i++) {
        float force_rate = (control.force[i] - prev_control.force[i]) / DT;
        float torque_rate = (control.torque[i] - prev_control.torque[i]) / DT;
        
        if (fabsf(force_rate) > max_rate) {
            control.force[i] = prev_control.force[i] + copysignf(max_rate * DT, force_rate);
        }
        
        if (fabsf(torque_rate) > max_rate) {
            control.torque[i] = prev_control.torque[i] + copysignf(max_rate * DT, torque_rate);
        }
    }
}

// Kernel for parallel control computation
__global__ void compute_control_kernel(
    const State* current_states,
    const Trajectory* trajectories,
    const Control* prev_controls,
    Control* controls,
    float max_force,
    float max_torque,
    float max_rate
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 1) return;  // Single controller for now
    
    // Get current state and trajectory
    const State& current_state = current_states[idx];
    const Trajectory& trajectory = trajectories[idx];
    const Control& prev_control = prev_controls[idx];
    
    // Compute lookahead state
    State lookahead_state;
    compute_lookahead_state(trajectory, current_state, lookahead_state);
    
    // Compute control
    Control control;
    compute_control(current_state, lookahead_state, control);
    
    // Apply constraints
    saturate_control(control, max_force, max_torque);
    limit_control_rate(control, prev_control, max_rate);
    
    // Store control
    controls[idx] = control;
}

// ROS2-exposed method
void compute_control(
    const State& current_state,
    const Trajectory& trajectory,
    const Control& prev_control,
    Control& control,
    float max_force = 100.0f,
    float max_torque = 50.0f,
    float max_rate = 10.0f
) {
    // Allocate device memory
    thrust::device_vector<State> d_current_states(1, current_state);
    thrust::device_vector<Trajectory> d_trajectories(1, trajectory);
    thrust::device_vector<Control> d_prev_controls(1, prev_control);
    thrust::device_vector<Control> d_controls(1);
    
    // Launch kernel for parallel control computation
    int block_size = 256;
    int num_blocks = (1 + block_size - 1) / block_size;
    compute_control_kernel<<<num_blocks, block_size>>>(
        thrust::raw_pointer_cast(d_current_states.data()),
        thrust::raw_pointer_cast(d_trajectories.data()),
        thrust::raw_pointer_cast(d_prev_controls.data()),
        thrust::raw_pointer_cast(d_controls.data()),
        max_force,
        max_torque,
        max_rate
    );
    
    // Copy control back to host
    control = d_controls[0];
} 