import argparse

def test_parser():
    parser = argparse.ArgumentParser(description="Runs test for Diff-DQN")

    parser.add_argument(
        '-t', '--test', 
        help="""
            Test number {1a, 1b, 2a, 2b, 3a, 3b}:
                1: Normal training
                1a: Fully connected network
                1b: Spoke network
                1c: Chain network

                2: Network partitions
                2a: Triangle network
                2b: Spoke network

                3: Learning through diffusion only
                3a: Spoke network
                3b: Chain network
            """, 
        required=True
    )

    parser.add_argument(
        '-l', '--length', 
        help="Number of episodes to run", 
        required=True
    )

    parser.add_argument(
        '-f', '--file', 
        help="Learned agent file for tests"
    )
    return parser