<div align="center">
  
# SlicerModelsVoxelization
<br>

<img src="https://raw.githubusercontent.com/DomenicoRiggio/SlicerModelsVoxelization/refs/heads/develop/Voxelization/Resources/Icons/Voxelization.png?raw=true" alt="drawing" style="width:300px;"/>



[![3D Slicer](https://img.shields.io/badge/3D_Slicer_-_5.8%20%7C%205.10%20%7C%205.11_Compatible-blue?style=for-the-badge)](https://download.slicer.org/)
</div>

---

A 3D Slicer extension for converting surface meshes and segmentations into solid cubical voxel models, with support for boolean operations, quantitative metrics, and multi-format export.

<div align="center">
  <img src="https://raw.githubusercontent.com/DomenicoRiggio/SlicerModelsVoxelization/refs/heads/develop/Voxelization/Resources/Icons/Examples.png?raw=true" alt="Examples" width="900"/>
</div>

---



## Features

### Voxelization
- Voxelize a **model node** or a **segmentation** (with per-segment selection) into a solid cubical mesh
- Adjustable **pitch** (voxel side length) with a user-defined maximum
- **Occupancy threshold**: keep only voxels covered by at least a given percentage of the mesh — removes partial-volume boundary voxels
- Results are automatically named `<input>_vox` and grouped in a **VoxelizedModels** folder in the Subject Hierarchy

### Boolean Operations
- Apply **Union**, **Intersection**, **Difference (A−B)**, or **Difference (B−A)** between any two voxelized models
- Boolean operations work best when both models are voxelized with the **same pitch**
- If the two models have different pitches, the operation is automatically performed at the **smaller pitch** — the coarser model is re-sampled accordingly and a warning message is shown

### Quantitative Metrics
- **Volume** of the original and voxelized model in cm³
- **ΔV** in both cm³ and %
- **Excluded voxel statistics**: mean ± std and median [IQR 5%–95%] of the occupancy percentage of voxels removed by the threshold — helps the user assess whether the threshold is appropriate
- Metrics are stored per model and can be reviewed by selecting any model from the dropdown
- **Export metrics as CSV** via a file save dialog

### Export
- Export any voxelized or boolean result in six formats:
  - **VTK** — ParaView, 3D Slicer, FEM tools
  - **STL** — 3D printing, any 3D software
  - **OBJ** — Blender, Maya, Unity (Gmsh-compatible, no comment headers)
  - **MSH** — Gmsh 2.2 ASCII format for FEM workflows
  - **PLY** — MeshLab, CloudCompare, open3d
  - **OFF** — MeshLab, geometry processing research tools

---

## Installation

1. Clone or download this repository
2. Open 3D Slicer
3. Go to **Edit → Application Settings → Modules**
4. Add the path to the `Voxelization` folder under **Additional module paths**
5. Restart 3D Slicer
6. The module appears under **Modules → Utilities → Voxelization**

On first load, missing Python dependencies (`trimesh`, `meshio`) are installed automatically with a confirmation dialog.

---

## Usage

### Basic workflow

1. Load a volume (MRI/CT) and a model or segmentation into the scene
2. Open the **Voxelization** module
3. Select **Input Type**: Segmentation or Model
4. Select the input segmentation/model, segment, and reference volume
5. Set **Pitch** (voxel size in cm³) and **Threshold** (occupancy filter, 0.0–1.0)
6. Click **Apply**
7. The voxelized result appears in the **VoxelizedModels** folder

### Boolean operations

1. Voxelize at least two models or segments
2. Expand **Boolean Operations**
3. Select **Model A**, **Model B**, and the **Operation**
4. Click **Apply Boolean Operation**

### Exporting

1. Expand **Export**
2. Select the model to export from the dropdown
3. Choose an export folder and format
4. Click **Export files to folder**

---

## Dependencies

- [3D Slicer](https://www.slicer.org/) 5.x
- [trimesh](https://trimsh.org/) — voxelization and mesh processing
- [meshio](https://github.com/nschloe/meshio) — MSH format export

Dependencies are installed automatically on first use.

---

## Contributors

- [Kimia Ghodousipour](https://github.com/kimiaghodoosi) (KIT Institute of Biomedical Engineering)
- Laura Lichtlein (KIT Institute of Biomedical Engineering)
- [Ciro Benito Raggio](https://github.com/ciroraggio/) (KIT Institute of Biomedical Engineering)
- [Domenico Riggio](https://github.com/DomenicoRiggio/) (KIT Institute of Biomedical Engineering)


