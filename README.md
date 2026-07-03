# Edge-Based Fetal ECG Signal Extraction

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)

## The Problem Space
Fetal electrocardiograms (FECG) provide critical data on fetal distress. However, maternal abdominal recordings are overwhelmingly contaminated by the maternal heartbeat and environmental noise. In rural or low-resource clinics, relying on cloud-compute to process these signals is impossible due to bandwidth constraints. 

This project solves this by using a lightweight 1D Convolutional Neural Network (CNN) to perform blind source separation, extracting the clean fetal heartbeat offline, entirely on the edge.

## Architecture & Methodology
The core model treats signal extraction as a continuous mapping problem. The 1D CNN relies on discrete-time convolution where the feature map $y[n]$ is defined by the input $x$ and kernel weights $w$:

$$y[n] = \sum_{k=0}^{K-1} x[n-k] w[k]$$

To achieve edge deployment without a backend server, the trained PyTorch model is quantized to INT8 precision and compiled to WebAssembly (Wasm), allowing it to run natively within a web browser.

## Repository Structure
* `/src` - 1D CNN architecture, training loops, and signal processing scripts.
* `/web` - WebAssembly deployment files and local browser frontend.
* `/data` - Reserved for the PhysioNet dataset (ignored in version control).

## Reproducibility
To run this project locally:

1. Clone the repository.
2. Install the requirements: `pip install -r requirements.txt`
3. Download the [PhysioNet Non-Invasive Fetal ECG Database](https://physionet.org/content/nifecgfo/) and place it inside the `/data` directory.
4. Run the training script via `python src/train.py` or launch the local web interface in `/web`.