#!/bin/bash

# Run several test
NUM_EPISODES=5000

for TRIAL in {1..20}
do
    echo "Trial $TRIAL"
    # 1: Normal training

    # 1a: Fully connected network
    python -m src.main -n $TRIAL -t "1a" -l $NUM_EPISODES 
    # 1b: Spoke network
    python -m src.main -n $TRIAL -t "1b" -l $NUM_EPISODES 
    # 1c: Chain network
    python -m src.main -n $TRIAL -t "1c" -l $NUM_EPISODES 


    # 2: Network partitions

    # 2a: Triangle network
    # python -m src.main -n $TRIAL -t "2a" -l $NUM_EPISODES 
    # 2b: Spoke network
    # python -m src.main -n $TRIAL -t "2b" -l $NUM_EPISODES 


    # 3: Learning through diffusion only

    # 3a: Spoke network
    python -m src.main -n $TRIAL -t "3a" -l 10000 
    # 3b: Chain network
    python -m src.main -n $TRIAL -t "3b" -l 10000 
done

echo "Finished Testing"