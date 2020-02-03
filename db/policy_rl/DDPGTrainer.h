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
#pragma once

#include <torch/torch.h>
#include <ExperienceReplay.h>

#include "OUNoise.h"
#include <Trainer.h>

class Actor : public torch::nn::Module {
  public:
    Actor(int64_t state_size, int64_t action_size, int64_t fc1_units=400, int64_t fc2_units=300);
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
    Critic(int64_t state_size, int64_t action_size, int64_t fcs1_units=400, int64_t fc2_units=300);
    void reset_parameters();
    torch::Tensor forward(torch::Tensor x, torch::Tensor action);
    std::pair<double,double> hidden_init(torch::nn::Linear& layer);

  private:
    torch::nn::Linear fcs1{nullptr}, fc2{nullptr}, fc3{nullptr};
    torch::nn::BatchNorm bn1{nullptr};
};  

class DDPGTrainer : public Trainer {
  public:
    double tau = 1e-3;              // for soft update of target parameters
    double lr_actor = 1e-4;         // learning rate of the actor
    double lr_critic = 1e-3;        // learning rate of the critic
    double weight_decay = 0;        // L2 weight decay
    
    OUNoise* noise;
    int64_t stateSize;
    int64_t actionSize;
    
    std::shared_ptr<Actor> actor_local;
    std::shared_ptr<Actor> actor_target;
    torch::optim::Adam actor_optimizer;

    std::shared_ptr<Critic> critic_local;
    std::shared_ptr<Critic> critic_target;
    torch::optim::Adam critic_optimizer;
       
  DDPGTrainer(int64_t input_channels, int64_t num_actions, int64_t capacity);
  std::vector<float> act(std::vector<float> state);
  void reset() {
    noise->reset();  
  }
  void learn();
  void soft_update(std::shared_ptr<torch::nn::Module> local, std::shared_ptr<torch::nn::Module> target);
  void hard_copy( std::shared_ptr<torch::nn::Module> local, std::shared_ptr<torch::nn::Module> target);
};

