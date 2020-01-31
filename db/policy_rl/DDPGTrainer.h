/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   DDPGTrainer.h
 * Author: wonki
 *
 * Created on January 31, 2020, 4:25 PM
 */

#include <torch/torch.h>

class DDPGTrainer : public Trainer {
  public:
    double TAU = 1e-3;              // for soft update of target parameters
    double LR_ACTOR = 1e-4;         // learning rate of the actor
    double LR_CRITIC = 1e-3;        // learning rate of the critic
    double WEIGHT_DECAY = 0;        // L2 weight decay
    
  DDPGTrainer(int64_t input_channels, int64_t num_actions, int64_t capacity);  
  
  class Actor : public torch::nn::Module {
    public:
      Actor(int64_t state_size, int64_t action_size, int64_t seed = 0, int64_t fc1_units=400, int64_t fc2_units=300);
      void reset_parameters();

      torch::Tensor forward(torch::Tensor state);
      torch::nn::BatchNormOptions bn_options(int64_t features);
      std::pair<double,double> hidden_init(torch::nn::Linear& layer);


    private:
      torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
      torch::nn::BatchNorm bn1{nullptr};
  };

  class Critic : public torch::nn::Module {
    public:
      Critic(int64_t state_size, int64_t action_size, int64_t seed = 0, int64_t fcs1_units=400, int64_t fc2_units=300);
      void reset_parameters();

      torch::Tensor forward(torch::Tensor x, torch::Tensor action);
      std::pair<double,double> hidden_init(torch::nn::Linear& layer);


    private:
      torch::nn::Linear fcs1{nullptr}, fc2{nullptr}, fc3{nullptr};
      torch::nn::BatchNorm bn1{nullptr};
  };    
};

