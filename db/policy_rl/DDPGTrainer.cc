/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
#include <math.h>
#include "DDPGTrainer.h"

/******************* Actor *******************/
Actor::Actor(int64_t state_size, int64_t action_size,
        int64_t fc1_units, int64_t fc2_units) : torch::nn::Module() {
  //torch::manual_seed(seed);
  fc1 = register_module("fc1", torch::nn::Linear(state_size, fc1_units));
  fc2 = register_module("fc2", torch::nn::Linear(fc1_units, fc2_units));
  fc3 = register_module("fc3", torch::nn::Linear(fc2_units, action_size));
//  bn1 = register_module("bn1", torch::nn::BatchNorm(fc1_units));
  reset_parameters();
}

std::pair<double,double> Actor::hidden_init(torch::nn::Linear& layer) {
  double lim = 1. / sqrt(layer->weight.sizes()[0]);
  return std::make_pair(-lim, lim);
}

void Actor::reset_parameters() {
  auto fc1_init = hidden_init(fc1);
  torch::nn::init::uniform_(fc1->weight, fc1_init.first, fc1_init.second);
  auto fc2_init = hidden_init(fc2);
  torch::nn::init::uniform_(fc2->weight, fc2_init.first, fc2_init.second);
  torch::nn::init::uniform_(fc3->weight, -3e-3, 3e-3);
}

torch::Tensor Actor::forward(torch::Tensor x) {
  x = torch::relu(fc1->forward(x));
//    bn1->forward(x);
  x = torch::relu(fc2->forward(x));
  x = fc3->forward(x);
  x = torch::tanh(x);
  return x;
}

torch::nn::BatchNormOptions Actor::bn_options(int64_t features) {
  torch::nn::BatchNormOptions bn_options = torch::nn::BatchNormOptions(features);
  bn_options.affine_ = true;
  bn_options.stateful_ = true;
  return bn_options;
}

/******************* Critic *****************/
Critic::Critic(int64_t state_size, int64_t action_size,
        int64_t fcs1_units, int64_t fc2_units) : torch::nn::Module() {
 // torch::manual_seed(seed);
  fcs1 = register_module("fcs1", torch::nn::Linear(state_size, fcs1_units));
  fc2 = register_module("fc2", torch::nn::Linear(fcs1_units + action_size, fc2_units));
  fc3 = register_module("fc3", torch::nn::Linear(fc2_units, 1));
//    bn1 = register_module("bn1", torch::nn::BatchNorm(fcs1_units));
  reset_parameters();
}

std::pair<double,double> Critic::hidden_init(torch::nn::Linear& layer) {
  double lim = 1. / sqrt(layer->weight.sizes()[0]);
  return std::make_pair(-lim, lim);
}

void Critic::reset_parameters() {
  auto fcs1_init = hidden_init(fcs1);
  torch::nn::init::uniform_(fcs1->weight, fcs1_init.first, fcs1_init.second);
  auto fc2_init = hidden_init(fc2);
  torch::nn::init::uniform_(fc2->weight, fc2_init.first, fc2_init.second);
  torch::nn::init::uniform_(fc3->weight, -3e-3, 3e-3);
}

torch::Tensor Critic::forward(torch::Tensor x, torch::Tensor action) {
  if (x.dim() == 1)
      x = torch::unsqueeze(x, 0);
   if (action.dim() == 1)
        action = torch::unsqueeze(action,0);
   auto xs = torch::relu(fcs1->forward(x));
//    xs = bn1->forward(xs);
  x = torch::cat({xs,action}, /*dim=*/1);
  x = torch::relu(fc2->forward(x));
  return fc3->forward(x);
}

DDPGTrainer::DDPGTrainer(int64_t input_channels, int64_t num_actions, int64_t capacity)
    :Trainer(capacity), actor_optimizer(actor_local->parameters(), lr_actor), critic_optimizer(critic_local->parameters(), lr_critic){
    stateSize = input_channels;
    actionSize = num_actions;
    
    actor_local = std::make_shared<Actor>(stateSize, actionSize);
    actor_target = std::make_shared<Actor>(stateSize, actionSize);
  
    critic_local = std::make_shared<Critic>(stateSize, actionSize);
    critic_target = std::make_shared<Critic>(stateSize, actionSize);

    actor_local->to(torch::Device(torch::kCPU));
    actor_target->to(torch::Device(torch::kCPU));

    critic_local->to(torch::Device(torch::kCPU));
    critic_target->to(torch::Device(torch::kCPU));

    critic_optimizer.options.weight_decay_ = weight_decay;

    hard_copy(actor_target, actor_local);
    hard_copy(critic_target, critic_local);
    noise = new OUNoise(static_cast<size_t>(actionSize));   
}  

std::vector<double> DDPGTrainer::act(std::vector<double> state) {
  torch::Tensor torchState = torch::tensor(state, torch::dtype(torch::kDouble)).to(torch::kCPU);
  actor_local->eval();

  torch::NoGradGuard guard;
  torch::Tensor action = actor_local->forward(torchState).to(torch::kCPU);
  actor_local->train();
  std::vector<double> v(action.data_ptr<double>(), action.data_ptr<double>() + action.numel());
  noise->sample(v);
  
  for (size_t i =0; i < v.size(); i++) {
    v[i] = std::fmin(std::fmax(v[i],-1.f), 1.f);
  }
  return v;
}

void DDPGTrainer::learn() {
  std::vector<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>> batch =
    buffer.sample_queue(batch_size);
  std::vector<torch::Tensor> states;
  std::vector<torch::Tensor> new_states;
  std::vector<torch::Tensor> actions;
  std::vector<torch::Tensor> rewards;

  for (auto i : batch) {
    states.push_back(std::get<0>(i));
    new_states.push_back(std::get<1>(i));
    actions.push_back(std::get<2>(i));
    rewards.push_back(std::get<3>(i));
  }

  torch::Tensor states_tensor = torch::cat(states, 0);
  torch::Tensor new_states_tensor = torch::cat(new_states, 0);
  torch::Tensor actions_tensor = torch::cat(actions, 0);
  torch::Tensor rewards_tensor = torch::cat(rewards, 0);
                    
  auto actions_next = actor_target->forward(new_states_tensor);
  auto Q_targets_next = critic_target->forward(new_states_tensor, actions_next);
  auto Q_targets = rewards_tensor + (gamma * Q_targets_next);
  auto Q_expected = critic_local->forward(states_tensor, actions_tensor); 

  torch::Tensor critic_loss = torch::mse_loss(Q_expected, Q_targets.detach());
  critic_optimizer.zero_grad();
  critic_loss.backward();
  critic_optimizer.step();

  auto actions_pred = actor_local->forward(states_tensor);
  auto actor_loss = -critic_local->forward(states_tensor, actions_pred).mean();

  actor_optimizer.zero_grad();
  actor_loss.backward();
  actor_optimizer.step();

  soft_update(critic_local, critic_target);
  soft_update(actor_local, actor_target);    
}

void DDPGTrainer::soft_update(std::shared_ptr<torch::nn::Module> local, std::shared_ptr<torch::nn::Module> target) {
  torch::NoGradGuard no_grad;
  for (size_t i = 0; i < target->parameters().size(); i++) {
    target->parameters()[i].copy_(tau * local->parameters()[i] + (1.0 - tau) * target->parameters()[i]);
  }   
}

void DDPGTrainer::hard_copy(std::shared_ptr<torch::nn::Module> local, std::shared_ptr<torch::nn::Module> target) {
  for (size_t i = 0; i < target->parameters().size(); i++) {
    target->parameters()[i] = local->parameters()[i];
  }
}

