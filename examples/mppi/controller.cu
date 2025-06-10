// This file is auto-generated from a .cuif specification
// Class: MPPIControllerNode
// Method: compute_control

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/iterator/zip_iterator.h>
#include <vector>
#include <cmath>

// Constants
constexpr int STATE_DIM = 12;  // 12-dimensional state vector
constexpr int CONTROL_DIM = 6;  // 6-dimensional control vector
constexpr int NUM_SAMPLES = 1000;  // Number of trajectory samples
constexpr float DT = 0.01f;  // Time step
constexpr int HORIZON = 50;  // Prediction horizon
constexpr float NOISE_SCALE = 0.1f;  // Noise scale for exploration
constexpr float TEMPERATURE = 1.0f;  // Temperature for softmax

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
    State states[HORIZON + 1];
    Control controls[HORIZON];
    float cost;
};

// Device functions for core computations
__device__ void compute_dynamics(State& next_state, const State& current_state, const Control& control) {
    // Position update
    for (int i = 0; i < 3; i++) {
        next_state.position[i] = current_state.position[i] + current_state.velocity[i] * DT;
        next_state.velocity[i] = current_state.velocity[i] + control.force[i] * DT;
    }
    
    // Orientation update (simplified)
    for (int i = 0; i < 3; i++) {
        next_state.orientation[i] = current_state.orientation[i] + current_state.angular_velocity[i] * DT;
        next_state.angular_velocity[i] = current_state.angular_velocity[i] + control.torque[i] * DT;
    }
}

__device__ float compute_state_cost(const State& state, const State& target_state) {
    float cost = 0.0f;
    
    // Position cost
    for (int i = 0; i < 3; i++) {
        float pos_diff = state.position[i] - target_state.position[i];
        cost += pos_diff * pos_diff;
    }
    
    // Velocity cost
    for (int i = 0; i < 3; i++) {
        cost += state.velocity[i] * state.velocity[i];
    }
    
    // Orientation cost
    for (int i = 0; i < 3; i++) {
        float ori_diff = state.orientation[i] - target_state.orientation[i];
        cost += ori_diff * ori_diff;
    }
    
    // Angular velocity cost
    for (int i = 0; i < 3; i++) {
        cost += state.angular_velocity[i] * state.angular_velocity[i];
    }
    
    return cost;
}

__device__ float compute_control_cost(const Control& control) {
    float cost = 0.0f;
    
    // Force cost
    for (int i = 0; i < 3; i++) {
        cost += control.force[i] * control.force[i];
    }
    
    // Torque cost
    for (int i = 0; i < 3; i++) {
        cost += control.torque[i] * control.torque[i];
    }
    
    return cost;
}

// Device function for trajectory cost computation
__device__ float compute_trajectory_cost(Trajectory& traj, const State& target_state) {
    float total_cost = 0.0f;
    
    // Compute cost for each state in trajectory
    for (int t = 0; t <= HORIZON; t++) {
        total_cost += compute_state_cost(traj.states[t], target_state);
    }
    
    // Add control costs
    for (int t = 0; t < HORIZON; t++) {
        total_cost += compute_control_cost(traj.controls[t]);
    }
    
    traj.cost = total_cost;
    return total_cost;
}

// Device function for noise sampling
__device__ void sample_noise(Control& noise, thrust::default_random_engine& rng) {
    thrust::normal_distribution<float> dist(0.0f, NOISE_SCALE);
    
    for (int i = 0; i < 3; i++) {
        noise.force[i] = dist(rng);
        noise.torque[i] = dist(rng);
    }
}

// Device function for control update
__device__ void update_control(Control& control, const Control& noise, float weight) {
    for (int i = 0; i < 3; i++) {
        control.force[i] += noise.force[i] * weight;
        control.torque[i] += noise.torque[i] * weight;
    }
}

// Device function for trajectory sampling
__device__ void sample_trajectory(Trajectory& traj, const State& initial_state, 
                                const Control* nominal_controls, thrust::default_random_engine& rng) {
    // Initialize trajectory
    traj.states[0] = initial_state;
    
    // Sample trajectory
    for (int t = 0; t < HORIZON; t++) {
        // Sample noise
        Control noise;
        sample_noise(noise, rng);
        
        // Update control with noise
        traj.controls[t] = nominal_controls[t];
        update_control(traj.controls[t], noise, 1.0f);
        
        // Compute next state
        compute_dynamics(traj.states[t + 1], traj.states[t], traj.controls[t]);
    }
}

// Device function for cost normalization
__device__ float normalize_cost(float cost, float min_cost, float max_cost) {
    return (cost - min_cost) / (max_cost - min_cost + 1e-6f);
}

// Device function for weight computation
__device__ float compute_weight(float normalized_cost) {
    return expf(-normalized_cost / TEMPERATURE);
}

// Kernel for parallel trajectory sampling
__global__ void sample_trajectories_kernel(Trajectory* trajectories, const State initial_state,
                                         const Control* nominal_controls, float* costs) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= NUM_SAMPLES) return;
    
    // Initialize random number generator
    thrust::default_random_engine rng;
    rng.discard(idx * HORIZON);
    
    // Sample trajectory
    sample_trajectory(trajectories[idx], initial_state, nominal_controls, rng);
    
    // Compute cost
    costs[idx] = compute_trajectory_cost(trajectories[idx], initial_state);
}

// Kernel for control update
__global__ void update_controls_kernel(Control* nominal_controls, const Trajectory* trajectories,
                                     const float* costs, float min_cost, float max_cost) {
    int t = blockIdx.x;
    int i = threadIdx.x;
    if (t >= HORIZON || i >= CONTROL_DIM) return;
    
    float total_weight = 0.0f;
    float weighted_update = 0.0f;
    
    // Compute weighted update for each trajectory
    for (int s = 0; s < NUM_SAMPLES; s++) {
        float normalized_cost = normalize_cost(costs[s], min_cost, max_cost);
        float weight = compute_weight(normalized_cost);
        total_weight += weight;
        
        if (i < 3) {
            weighted_update += weight * trajectories[s].controls[t].force[i];
        } else {
            weighted_update += weight * trajectories[s].controls[t].torque[i - 3];
        }
    }
    
    // Update nominal control
    if (total_weight > 0.0f) {
        if (i < 3) {
            nominal_controls[t].force[i] = weighted_update / total_weight;
        } else {
            nominal_controls[t].torque[i - 3] = weighted_update / total_weight;
        }
    }
}

// ROS2-exposed method
void compute_control(const State& current_state, const State& target_state, Control& control) {
    // Allocate device memory
    thrust::device_vector<Trajectory> d_trajectories(NUM_SAMPLES);
    thrust::device_vector<Control> d_nominal_controls(HORIZON);
    thrust::device_vector<float> d_costs(NUM_SAMPLES);
    
    // Initialize nominal controls
    for (int t = 0; t < HORIZON; t++) {
        for (int i = 0; i < 3; i++) {
            d_nominal_controls[t].force[i] = 0.0f;
            d_nominal_controls[t].torque[i] = 0.0f;
        }
    }
    
    // Launch kernel for parallel trajectory sampling
    int block_size = 256;
    int num_blocks = (NUM_SAMPLES + block_size - 1) / block_size;
    sample_trajectories_kernel<<<num_blocks, block_size>>>(
        thrust::raw_pointer_cast(d_trajectories.data()),
        current_state,
        thrust::raw_pointer_cast(d_nominal_controls.data()),
        thrust::raw_pointer_cast(d_costs.data())
    );
    
    // Find min and max costs
    float min_cost = thrust::reduce(d_costs.begin(), d_costs.end(), FLT_MAX, thrust::minimum<float>());
    float max_cost = thrust::reduce(d_costs.begin(), d_costs.end(), -FLT_MAX, thrust::maximum<float>());
    
    // Launch kernel for control update
    update_controls_kernel<<<HORIZON, CONTROL_DIM>>>(
        thrust::raw_pointer_cast(d_nominal_controls.data()),
        thrust::raw_pointer_cast(d_trajectories.data()),
        thrust::raw_pointer_cast(d_costs.data()),
        min_cost,
        max_cost
    );
    
    // Copy first control to output
    control = d_nominal_controls[0];
} 