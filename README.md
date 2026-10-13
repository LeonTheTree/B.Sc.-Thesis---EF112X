# Performance Analysis of Graph Neural Networks (GCNs)
*🏆 **Official Publication:** This thesis is officially published on the Swedish Digital Scientific Archive (DiVA) [Read the full publication on DiVA here](https://kth.diva-portal.org/smash/record.jsf?pid=diva2:2034028).*

This repository contains the code and report for my B.Sc. Thesis project (EF112X) at KTH Royal Institute of Technology. The project evaluates the performance of Graph Convolutional Networks (GCNs) on the Cora citation dataset, specifically analyzing the impact of varying model depth, dropout rates, and optimizer choices.

## Personal Contributions
In this collaborative project, my primary responsibilities included **researching the underlying GNN theory, setting up the experimental frameworks, and conducting the final performance analysis**. 

## Project Structure
The entire experimental pipeline and codebase for this project is self-contained within a single Jupyter Notebook. The repository also includes the final project report and appendix for detailed theoretical background, mathematical formulations, and expanded analysis.

## 🛠 Requirements
The project uses `uv` for efficient dependency management. The notebook is designed to automatically detect your system architecture (Apple Silicon MPS, CUDA, or CPU) and install the hardware-appropriate PyTorch and PyG versions. 

Core dependencies include:
* `torch`, `torchvision`, `torchaudio`
* `torch-geometric` (PyG)
* `numpy`, `scipy`
* `matplotlib`, `seaborn`
* `scikit-learn`, `networkx`, `umap-learn`

## 🚀 How to Run
1. Clone this repository.
2. Open the Jupyter Notebook.
3. Run all cells in sequential order. 

The notebook will automatically handle the installation of dependencies, download the Cora dataset via Planetoid, and execute the base model training alongside the three primary hyperparameter experiments.

## 📊 Key Findings
* **Model Depth:** Increasing the number of message-passing layers negatively impacted performance, likely due to over-smoothing (reducing discriminative power as node features homogenize).
* **Dropout & Optimizers:** Varying dropout rates showed minimal variance in final accuracy on this specific dataset. While different optimizers reached similar final accuracies, advanced adaptive optimizers demonstrated significantly faster convergence than standard SGD. 
* **Overall Performance:** The best-performing model configuration achieved a test accuracy of 81.9%, which is highly competitive for the relatively simple approach used on the Cora Planetoid split.
