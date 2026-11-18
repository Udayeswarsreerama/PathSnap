#  PathSnap — Distributed Map Matching & Route Reconstruction

## Overview

PathSnap is a machine learning-based GPS map-matching and route reconstruction system designed to reconstruct realistic vehicle trajectories from noisy GPS observations.

GPS tracking systems can produce inaccurate location measurements due to urban environments, signal interference, multipath effects, GPS dropouts, and low sampling frequencies. These inaccuracies can cause vehicles to appear on parallel roads, jump across buildings, or follow unrealistic routes.

PathSnap aims to address these challenges by matching GPS observations to a road network and reconstructing the most probable continuous route using machine learning and sequence-based decoding techniques.

---

## Problem Statement

Raw GPS trajectories do not always accurately represent the roads traveled by vehicles. Noise, missing observations, and low sampling rates make it difficult to identify the correct road segments and reconstruct the actual route.

The objective of PathSnap is to process GPS trajectories, identify suitable road candidates, predict road matches, and reconstruct a geographically consistent route.

---

## Objectives

- Process and analyze raw GPS trajectory data.
- Handle noisy and irregular GPS observations.
- Generate candidate road segments for GPS points.
- Extract useful features for road matching.
- Train and compare multiple machine learning models.
- Perform Bayesian hyperparameter optimization.
- Apply Hidden Markov Model (HMM) and Viterbi decoding for sequence-based route reconstruction.
- Evaluate map-matching and route-reconstruction performance.
- Provide a user interface for selecting trips and visualizing reconstructed routes.
- Support distributed execution through a backend server and a separate evaluator-facing interface.

---

## Proposed Methodology

The PathSnap pipeline consists of the following stages:

1. **Data Exploration** — Analyze the trajectory dataset, road network, and available route information.
2. **Preprocessing** — Clean and organize GPS observations and prepare trajectory data.
3. **Candidate Generation** — Identify nearby road segments that could correspond to each GPS observation.
4. **Feature Engineering** — Extract spatial, temporal, distance, direction, and road-related features.
5. **Model Training** — Train and compare multiple machine learning models.
6. **Hyperparameter Optimization** — Improve the selected model using Bayesian optimization.
7. **Sequence Decoding** — Apply HMM and Viterbi decoding to identify a probable road sequence.
8. **Route Reconstruction** — Produce a continuous road-matched trajectory.
9. **Evaluation** — Analyze prediction performance and route reconstruction quality.
10. **Visualization** — Display GPS observations and reconstructed routes through an interactive map interface.

---

## System Architecture

The system is designed around two separate machines:

### Laptop 1 — ML Backend Server

Responsible for:

- Dataset processing
- Candidate road generation
- Feature extraction
- Machine learning inference
- HMM and Viterbi decoding
- Route reconstruction
- Evaluation and prediction response generation
- Hosting the API server

### Laptop 2 — Evaluator Interface

Responsible for:

- Trip selection
- Sending prediction requests to the backend
- Displaying processing status
- Visualizing GPS trajectories
- Displaying reconstructed routes
- Presenting available evaluation metrics and results

The two components communicate through HTTP API requests.

---

## Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Machine Learning | Scikit-learn and selected ML libraries |
| Data Processing | Pandas, NumPy |
| Route Reconstruction | HMM, Viterbi |
| Backend API | FastAPI |
| Frontend | Streamlit |
| Visualization | Folium, Matplotlib |
| Hyperparameter Optimization | Bayesian Optimization framework |
| Version Control | Git, GitHub |

*The technology stack may be updated during development based on implementation requirements.*

---

## Repository Structure

```text
PathSnap/
├── data/
├──
