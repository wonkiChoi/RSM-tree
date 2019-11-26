//
// Created by Navneet Madhu Kumar on 2019-07-12.
//
#pragma once

#include <torch/torch.h>
#include "ExperienceReplay.h"
#include "dqn.h"

class Trainer {
  public:
    ExperienceReplay buffer;
    DQN network, target_network;
    torch::optim::Adam dqn_optimizer;
    double epsilon_start = 0.8;
    double epsilon_final = 0.01;
    int64_t epsilon_decay = 30000;
    int64_t batch_size = 32;
    float gamma = 0.99;
    int64_t frame_id;
    int64_t previous_action;
    std::vector<int64_t> state;
    torch::Tensor state_tensor;
    int64_t num_level;
    int64_t max_file_num;
  
    Trainer(int64_t input_channels, int64_t num_actions, int64_t capacity, int64_t frame_id_, int64_t previous_action_);
    torch::Tensor compute_td_loss(int64_t batch_size, float gamma);
    double epsilon_by_frame();
    torch::Tensor get_tensor_observation(std::vector<int64_t> state);
    void loadstatedict(torch::nn::Module& model,
                       torch::nn::Module& target_model);

};
