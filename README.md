# FPGA-Based Drone Detection and Classification System Using YOLOv8n on ZCU104

## Overview

This project implements a real-time FPGA-based drone detection and classification system using YOLOv8n deployed on the AMD Xilinx ZCU104 platform.

The system combines deep learning acceleration using the DPUCZDX8G accelerator with a custom hardware Non-Maximum Suppression (NMS) IP developed in Vivado HLS.

---

## Key Features

* Real-time drone detection
* YOLOv8n object detection network
* INT8 quantization using Vitis AI
* Deployment on ZCU104 FPGA platform
* Custom NMS Accelerator using Vivado HLS
* AXI4-Stream based hardware integration
* Reduced CPU overhead through hardware acceleration

---

## Hardware Platform

* AMD Xilinx ZCU104
* DPUCZDX8G Accelerator
* ARM Cortex-A53 Processing System
* Vivado Design Suite
* Vitis AI

---

## Design Flow

Dataset → YOLOv8 Training → ONNX Export → INT8 Quantization → DPU Compilation → FPGA Deployment → Custom NMS Accelerator → Detection Output

---

## Project Architecture

See architecture diagrams in the images folder.

---

## Performance

### Quantized Model Performance

| Metric       | Value |
| ------------ | ----- |
| Precision    | 91.4% |
| Recall       | 91.6% |
| mAP@0.5      | 93.8% |
| mAP@0.5:0.95 | 54.9% |

---

## Custom NMS Accelerator

The NMS module was implemented using Vivado HLS and integrated into the FPGA design through AXI4-Stream interfaces.

Files:

* hls/nms_stream.cpp
* hls/tb_nms_stream.cpp

---

## Repository Structure

docs/ → Project documents

images/ → Architecture and results

hls/ → Custom NMS accelerator source code

results/ → Experimental results

---

## Author

M. Surya Vardhan

M.Tech VLSI Design

VIT Chennai
