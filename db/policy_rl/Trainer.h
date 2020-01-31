/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   Trainer.h
 * Author: wonki
 *
 * Created on January 31, 2020, 4:58 PM
 */

class Trainer {
  public:
    int64_t batch_size = 32;
    float gamma = 0.99;
    int64_t frame_id = 0;
    int64_t previous_action = 0;
    int64_t input_channels;
    int64_t num_actions;
    std::vector<float> PrevState;
    torch::Tensor PrevStateTensor;
    std::vector<float> NewState; 
  
    Trainer(int64_t input_channels, int64_t num_actions, int64_t capacity) = 0;
};