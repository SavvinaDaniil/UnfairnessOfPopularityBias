# UnfairnessOfPopularityBias

![Python Version](https://img.shields.io/badge/python-3.8-blue)

Welcome to Unfairness of Popularity Bias! In this repository, we are working on reproducing three studies on the topic of unfairness of popularity bias, namely Abdollahpouri et al., [2019](https://arxiv.org/abs/1907.13286), Kowald et al., [2019](https://link.springer.com/chapter/10.1007/978-3-030-45442-5_5), and Naghiaei et al., [2022](https://arxiv.org/abs/2202.13446). At the same time, we are experimenting with various aspects of the recommendation and evaluation process that vary across the three studies, specifically:

* **Data**: the studies use three datasets, with different characteristics such as size, sparsity, and distribution of item popularity.
* **Algorithms**: the studies evaluate mostly different algorithms, with some excpetions.
* **Division** of users in groups: the studies define propensity for popular items differently and divide them accordingly.
* **Evaluation** strategy: the studies make different choices in the testing process.

## Table of Contents

- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)

## Getting Started

To get started with this project, follow the instructions below.

### Prerequisites

Make sure you have the following software installed:

- Python 3.8

## Installation

1. Create and activate a conda virtual environment named:

   ```bash
   conda create --name tors python=3.8
   conda activate tors
   ```

2. Clone the repository:

   ```bash
   git clone https://github.com/SavvinaDaniil/UnfairnessOfPopularityBias.git
   cd UnfairnessOfPopularityBias
   ```

3. Install the required Python packages:

   ```bash
   pip install -r requirements.txt
   ```
## Usage

Once you have completed the installation steps, you can now run the experiments using the provided Jupyter notebooks.

1. Start the Jupyter notebook server:

   ```bash
   
   jupyter notebook
   ```
   Make sure that jupyter points to the jupyter installed from the requirements file. You may need to deactivate and activate the environment again.

2. Open the recommendation notebooks from the project directory.

