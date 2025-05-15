# Master Thesis Damian Gerber: Learning to Route on Surfaces, a geodesic NCO model for the Montreal CVRP in 3D

This repository contains the implementation of Damian Gerbers Master's thesis at the Massachusetts Institute of Technology (MIT), conducted at the Center for Transportation and Logistics (CTL) in the Intelligent Logistics Systems Lab under the supervision of Prof. Matthias Winkenbach.

## Project Overview

This thesis explores the application of geometric and deep learning techniques to solve vehicle routing problems on arbitrary 3D surfaces.

### 1. Manifold Learning for Urban Data (Early Work)
Located in the `manifold_learning/` directory, this component contains the initial exploration of processing real-world city data using manifold learning techniques and Multi-Dimensional Scaling (MDS). This work laid the foundation for understanding how to represent and process geometric data from urban environments. This code is not relevant but left if used in the future.

### 2. GEHD Model Implementation
The main contribution of this thesis is the Geometric Encoder and Heavy Decoder (GEHD) model, implemented in the `main_model/` directory. This model represents a novel approach to solving Capacitated Vehicle Routing Problems (CVRP) in 3D on arbitrary surfaces using 3D meshes.

For detailed information about the GEHD model implementation, please refer to the [main_model README](main_model/README.md).

## Repository Structure

```
.
├── main_model/           # Implementation of the GEHD model
│   ├── configs/         # Configuration files
│   ├── src/            # Source code
│   └── requirements.txt # Python dependencies
├── manifold_learning/   # Early work on manifold learning
├── .gitignore
├── pyproject.toml   # Project configuration and dependencies
└── README.md           # This file
```

## Contact

**Damian Gerber**
- Email: [dagerber@mit.edu](mailto:dagerber@mit.edu)
- Alternative Email: [damian@turbokids.com](mailto:damian@turbokids.com)
