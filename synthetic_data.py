"""
Simple script to generate synthetic data from a Bayesian network.
"""

from pgmpy.readwrite import BIFReader
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
import numpy as np
import pandas as pd


def generate_data_sample(bayesian_network: BayesianNetwork):
    # Iterate through the network topologically. We'll get nodes in a suitable order.
    sample = {}

    for node in bayesian_network.nodes():
        parents = bayesian_network.get_parents(node)
        cpt = bayesian_network.get_cpds(node)

        if parents:
            # Sample based on the CPT and parent values
            state_names = {
                parent: sample[parent] for parent in parents
            }  # Get states of parents
            cpt = bayesian_network.get_cpds(node).to_factor()
            probabilities = []
            for value in cpt.state_names[node]:
                probabilities.append(cpt.get_value(**state_names, **{node: value}))
            sample[node] = np.random.choice(cpt.state_names[node], p=probabilities)

        else:
            # Sample directly from the node's CPT
            cpt = bayesian_network.get_cpds(node)
            probabilities = cpt.get_values().flatten()
            sample[node] = np.random.choice(cpt.state_names[node], p=probabilities)

    return sample


dataset_name = "asia"
path = f"C:\\Users\\Luca\\Desktop\\MagistraleAI\\1st_year\\FundamentalsOfAI\\Module3\\Project\\data\\{dataset_name}.bif"
num_samples = 10000

reader = BIFReader(path)
model = reader.get_model()

dataset = [generate_data_sample(model) for _ in range(num_samples)]  # Generate samples

# Convert to a Pandas DataFrame
df = pd.DataFrame(dataset)

# Save the dataset to a CSV file
df.to_csv(f"./data/{dataset_name}.csv", index=False)
