Implementation Guide for Enhanced Matrix Inference with Seq2seq Models via Diagonal Sorting
This guide provides detailed instructions to implement the enhanced matrix inference technique using sequence-to-sequence (seq2seq) models with diagonal sorting, as described in the 2024 research paper by Peng, Wang, and Wu. The method improves seq2seq model performance on matrix-related tasks by introducing a diagonal-based sorting (DS) strategy, particularly effective for matrices of varying ranks. The implementation is available in a GitHub repository, and this guide covers setup, execution, and customization.
Background
Sequence-to-sequence (seq2seq) models, traditionally used for tasks like machine translation, have been adapted for mathematical and matrix-related tasks. However, standard seq2seq models struggle with matrices of unseen ranks (sizes) when using generic sorting methods like row-based or column-based sorting. The diagonal sorting method addresses this by:

Dividing a matrix into sub-blocks based on the main diagonal.
Sorting elements within each sub-block consistently.
Concatenating these sub-blocks to form a sequence that preserves structural properties across different matrix sizes.

This approach ensures "mapping invariance," meaning the ordering of elements remains consistent for leading principal submatrices, enhancing the model's ability to generalize to new matrix sizes. The technique has been validated on tasks like matrix transposition, finding maximum independent sets in graphs, and solving Sudoku puzzles.
Prerequisites
To implement this technique, you need the following:

Hardware:
A computer with at least 8 GB RAM.
Optional: A GPU (e.g., NVIDIA Tesla V100-SXM2 with 32 GB memory) for faster training, though a CPU is sufficient.


Software:
Python 3.8.
PyTorch 1.10.
NetworkX 2.8.
Additional libraries specified in the repository’s requirements.txt.


Operating System: The original experiments used Debian 8.3.0, but Windows, macOS, or other Linux distributions should work with proper setup.
Skills: Basic knowledge of Python, command-line operations, and machine learning concepts (e.g., Transformers).

Step-by-Step Implementation
1. Clone the GitHub Repository
The code is hosted in the ds-for-seq2seq repository. Clone it to your local machine using Git:
git clone https://github.com/Peng-weil/ds-for-seq2seq.git
cd ds-for-seq2seq

This repository contains:

Scripts for dataset generation.
The main script (main.py) for training and testing the seq2seq model.
Configuration files (requirements.txt, freeze.yml) for environment setup.
Pre-trained models and datasets (or instructions to generate them).

2. Set Up the Environment
The repository provides two configuration files:

requirements.txt: Lists Python dependencies.
freeze.yml: Specifies exact dependency versions for reproducibility.

To install dependencies using requirements.txt:
pip install -r requirements.txt

If you prefer exact version control, use freeze.yml with a tool like conda:
conda env create -f freeze.yml
conda activate ds-for-seq2seq

Ensure PyTorch 1.10 and NetworkX 2.8 are installed, as they are critical for the Transformer model and graph-related tasks (e.g., maximum independent set).
3. Obtain the Dataset
You can either generate the dataset using provided scripts or download pre-generated data.

Option 1: Generate the Dataset

Check the repository’s README or dataset generation scripts (likely in a dataset/ directory).
Run the appropriate script to create datasets for tasks like maximum independent set (MIS) or Sudoku. For example:python generate_dataset.py


The exact command depends on the repository’s structure, so refer to the documentation.


Option 2: Download the Dataset

The pre-generated dataset is available on Google Drive: MIS and Sudoku Dataset.
Download the dataset and place it in the ./dataset/ directory (create the directory if it doesn’t exist):mkdir dataset
mv path/to/downloaded/dataset/* ./dataset/





The dataset includes:

Maximum Independent Set (MIS): Pairs of adjacency matrices and their maximum independent sets (e.g., 200,000 samples for 20-node graphs).
Sudoku: 100,000 9×9 Sudoku puzzles with solutions.
Transposition: Matrix pairs for testing rank generalization.

4. Run the Experiments
The repository provides example commands to run experiments. For the Maximum Independent Set (MIS) task, use:
python main.py --exp_name "mis" --emb_dim 256 --n_enc_layers 6 --n_dec_layers 6 --n_heads 8 --batch_size 64 --reload_data "./dataset/MIS/200000_20_20_e2632" --reload_testset "{'20 nodes': './dataset/MIS/200000_20_20_e2632','19 nodes': './dataset/MIS/200000_19_19_hdzpw','18 nodes': './dataset/MIS/200000_18_18_5r9jy','17 nodes': './dataset/MIS/200000_17_17_fo19y','16 nodes': './dataset/MIS/200000_16_16_4v364'}" --sort_method "SMD,SMR"

Command Breakdown



Parameter
Description



--exp_name "mis"
Specifies the experiment (e.g., "mis" for Maximum Independent Set).


--emb_dim 256
Sets the embedding dimension to 256.


--n_enc_layers 6
Uses 6 encoder layers in the Transformer.


--n_dec_layers 6
Uses 6 decoder layers in the Transformer.


--n_heads 8
Sets 8 attention heads in the Transformer.


--batch_size 64
Sets the batch size for training.


--reload_data
Path to the training dataset.


--reload_testset
Dictionary of test datasets for different node counts (for MIS).


--sort_method
Compares sorting methods: SMD (diagonal), SMR (row-based), SMC (column-based), c-SMD (counter-diagonal).


Sorting Methods



Method
Description



SMR
Row-based sorting: Orders elements by rows.


SMC
Column-based sorting: Orders elements by columns.


SMD
Diagonal-based sorting: Sorts sub-blocks along the main diagonal (proposed method).


c-SMD
Counter-diagonal sorting: Sorts sub-blocks in reverse along the diagonal.


To run other experiments (e.g., Sudoku or transposition), modify the --exp_name and dataset paths according to the repository’s instructions.
5. Understand the Diagonal Sorting Implementation
The diagonal sorting (DS) method is the core innovation. It works as follows:

Divide the Matrix:

A matrix is split into sub-blocks based on the main diagonal elements (inspired by Cantor’s diagonal argument).
For a matrix ( \mathbf{M} ), sub-blocks are defined as:
( S^1_{\mathbf{M}} = [\mathtt{m_{1,1}}] )
( S^2_{\mathbf{M}} = [\mathtt{m_{1,2}, m_{2,2}, m_{2,1}}] )
And so on, up to ( S^n_{\mathbf{M}} ).




Sort Within Sub-Blocks:

Elements within each sub-block are sorted in a consistent direction (e.g., row-major or column-major within the block).


Concatenate Sub-Blocks:

The final sequence is formed by concatenating sub-block sequences:[S_{\mathbf{M}} = S^1_{\mathbf{M}} \oplus S^2_{\mathbf{M}} \oplus \cdots \oplus S^n_{\mathbf{M}}]
This ensures mapping invariance, meaning smaller submatrices maintain consistent ordering.



The counter-diagonal sorting (c-DS) variant reverses the sorting direction within sub-blocks but follows the same structure.
In the code, look for functions or modules in main.py or related scripts that implement this logic, likely in the data preprocessing or matrix-to-sequence conversion steps.
6. Verify Results
The paper reports results for three tasks, which you can replicate:



Task
Dataset Details
Metrics
Expected Outcome



Transposition
1000 pairs of 20/30-rank matrices
Single element accuracy ((\text{ACC}{\text{single}})), Full matrix accuracy ((\text{ACC}{\text{total}}))
DS/c-DS outperform SMR/SMC for unseen ranks.


Maximum Independent Set
200K (16-node), 400K (25-node) graphs
Accuracy on reduced node counts
DS/c-DS maintain higher accuracy as nodes decrease.


Sudoku
100K 9×9 puzzles, 8:2 train-test split
Single element accuracy, Full puzzle accuracy
DS/c-DS achieve higher full puzzle accuracy.


After running the experiments, compare your results with the paper’s tables (Tables 1–3 in the original publication).
7. Customize and Extend
To adapt the code for other tasks:

Modify the Dataset: Create new matrix-based datasets for your task and adjust the preprocessing to use diagonal sorting.
Tune Hyperparameters: Experiment with different embedding dimensions, layers, or attention heads in the command-line arguments.
Change the Model: The code uses a Transformer, but you could modify it to use an RNN-based seq2seq model (e.g., LSTM) by altering the model architecture in the source code.

Troubleshooting

Dependency Issues: If pip install fails, ensure Python 3.8 and compatible PyTorch versions are used. Check the PyTorch website for installation instructions: PyTorch.
Dataset Errors: Verify that dataset paths in the command match the actual file locations.
Performance: If training is slow on a CPU, reduce the batch size (e.g., --batch_size 32) or use a GPU.
Code Questions: If the repository’s README is unclear, check the paper for context or explore the source code for comments.

Citation
If you use this code or method, cite the original paper:

Peng, W., Wang, Y. & Wu, M. Enhanced matrix inference with Seq2seq models via diagonal sorting. Scientific Reports, 14, 883 (2024). https://doi.org/10.1038/s41598-023-50919-2

Conclusion
This implementation enables you to explore enhanced matrix inference using seq2seq models with diagonal sorting. By following the steps above, you can set up the environment, run the experiments, and potentially extend the method to new tasks. The GitHub repository provides all necessary resources, and the paper offers detailed insights into the method’s design and performance.
