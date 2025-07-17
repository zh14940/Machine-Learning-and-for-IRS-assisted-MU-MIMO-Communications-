# Machine Learning and Deep Learning for IRS-assisted MU-MIMO Communications

This repository contains the source code for two research directions in IRS-assisted multi-user MIMO systems:

1. **Machine Learning for IRS-assisted MU-MIMO Communications with Estimated Channels**
2. **CNN-enabled Joint Active and Passive Beamforming for RIS-assisted MU-MIMO Systems**

---

## 📌 Project Summary

This codebase explores data-driven solutions for improving downlink performance in **reconfigurable intelligent surface (RIS)-assisted multi-user MIMO (MU-MIMO)** wireless systems.

- The **first part** implements classical machine learning models for downlink power allocation and scheduling based on **estimated channel state information (CSI)**.
- The **second part** investigates a **convolutional neural network (CNN)-based framework** for **joint active and passive beamforming**, aiming to optimize both BS and RIS configurations in a unified learning pipeline.

---

## 🛰️ Channel Data Source (CSI Generation)

**Channel state information (CSI)** is not provided directly in this repository.  
You are expected to generate your own CSI dataset using the [**DeepMIMO**](https://deepmimo.net/) dataset generator.

### ▶️ How to Generate CSI:

1. Visit [https://www.deepmimo.net/](https://www.deepmimo.net/)
2. Download the **DeepMIMO Dataset Generator**
3. Configure the scenario file (e.g., `I1` or `O1`) with parameters matching your RIS-assisted MU-MIMO system
4. Run the generator to export channel matrices
5. Save the generated dataset in the `data/` directory of this repository