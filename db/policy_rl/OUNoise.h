/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   agent.h
 * Author: wonki
 *
 * Created on January 31, 2020, 4:20 PM
 */

class OUNoise {
private:
  size_t size;
  std::vector<float> mu;
  std::vector<float> state;
  float theta=0.15;
  float sigma=0.1;

public:
  OUNoise (size_t size_in) {
    size = size_in;
    mu = std::vector<float>(size, 0);
    reset();
  }

  void reset() {
    state = mu;
  }

  std::vector<float> sample(std::vector<float> action) {
    for (size_t i = 0; i < state.size(); i++) {
      auto random = ((float) rand() / (RAND_MAX));
      float dx = theta * (mu[i] - state[i]) + sigma * random;
      state[i] = state[i] + dx;
      action[i] += state[i];
    }
    return action;
 }
};