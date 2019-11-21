#!/bin/bash

# Run several test
NUM_EPISODES=2000

# 1: Normal training
echo "Starting Test suite 1..."

# 1a: Fully connected network
echo "Test 1a"
# python -m src.main -t "1a" -l $NUM_EPISODES 
# 1b: Spoke network
echo "Test 1b"
# python -m src.main -t "1b" -l $NUM_EPISODES 
# 1c: Chain network
echo "Test 1c"
python -m src.main -t "1c" -l $NUM_EPISODES 

echo "Test suite 1 completed"

# 2: Network partitions
echo "Starting Test suite 2..."

# 2a: Triangle network
echo "Test 2a"
python -m src.main -t "2a" -l $NUM_EPISODES 
# 2b: Spoke network
echo "Test 2b"
python -m src.main -t "2b" -l $NUM_EPISODES 

echo "Test suite 2 completed"

# 3: Learning through diffusion only
echo "Starting Test suite 3..."

# 3a: Spoke network
echo "Test 3a"
python -m src.main -t "3b" -l $NUM_EPISODES 
# 3b: Chain network
echo "Test 3b"
python -m src.main -t "3b" -l $NUM_EPISODES 