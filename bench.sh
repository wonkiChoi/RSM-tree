#!/bin/bash

./db_bench -db=/home/wonki/data_rocksdb --benchmarks="fillrandom,stats" -compression_type=none -num_levels=5 -key_size=8 -value_size=4096 -num=100000000 > kRSM.txt
