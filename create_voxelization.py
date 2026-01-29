import os
import sys
import numpy as np
import trimesh
import vtk
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox

def load_mesh_any(file_path):
    """
    Defines a function that loads a mesh from .stl .obj .vtk
    
    param file_path: location of the file containing the mesh
    """

    # extract file extension to decide which file loader to use
    ext = os.path.splitext(file_path)[1].lower()

    # load .stl or .obj mesh
    if ext in [".stl", ".obj"]:
        mesh = trimesh.load(file_path)
        if not isinstance(mesh, trimesh.Trimesh) and hasattr(mesh, "geometry"):
            mesh = list(mesh.geometry.values())[0]
        return mesh

    # vtk file reader
    elif ext == ".vtk":
        reader = vtk.vtkGenericDataObjectReader()
        reader.SetFileName(file_path)
        reader.Update() # executes the read
        output = reader.GetOutput() # retrieves the VTK dataset

        # if retrieved dataset is an unstructured grid = not a surface polydata, then convert it to surface geometry
        if isinstance(output, vtk.vtkUnstructuredGrid):
            geo = vtk.vtkGeometryFilter()
            geo.SetInputData(output)
            geo.Update()
            output = geo.GetOutput()

        # builds a NumPy array of point coordinates by iterating over VTK points
        points = np.array([output.GetPoint(i) for i in range(output.GetNumberOfPoints())])

        # Iterates through VTK cells and collects triangular faces
        faces = []
        for i in range(output.GetNumberOfCells()):
            cell = output.GetCell(i)
            if cell.GetNumberOfPoints() == 3:
                faces.append([cell.GetPointId(j) for j in range(3)])
        # store the faces as an Nx3 integer array
        faces = np.array(faces)

        # Constructs and returns a trimesh.Trimesh object from the points and faces extracted
        return trimesh.Trimesh(vertices=points, faces=faces)

    # if the file extension is something other than .obj .stl .vtk raise an error
    else:
        raise ValueError(f"Unsupported file type: {ext}")

def save_voxel_mesh_as_vtk(voxels, pitch, origin, out_path):
    """Save filled voxels as ASCII .vtk (legacy) with hexahedral cells, Gmsh-compatible."""

    # Unpacks the voxel grid shape (z, y, x axes)
    # Note: trimesh gives matrix as (z, y, x)
    nz, ny, nx = voxels.shape

    # Empty list to collect all cube corner coordinates, it will contain duplicates before uniquing
    points = []

    # Empty list to collect per-voxel connectivity, indices into points
    cells = []

    print("🧩 Building voxel hexahedral mesh as VTK file...")

    # Each voxel = cube = 8 points + 1 cell
    # iterate every voxel index (z, y, x)
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):

                # Check if that voxel is filled = True = 1
                if voxels[z, y, x]:

                    # compute world coordinates of the voxel minimum corner using origin and pitch
                    x0 = origin[0] + x * pitch
                    y0 = origin[1] + y * pitch
                    z0 = origin[2] + z * pitch

                    # lists the 8 corner coordinates in a consistent ordering
                    cube = [
                        [x0, y0, z0],
                        [x0 + pitch, y0, z0],
                        [x0 + pitch, y0 + pitch, z0],
                        [x0, y0 + pitch, z0],
                        [x0, y0, z0 + pitch],
                        [x0 + pitch, y0, z0 + pitch],
                        [x0 + pitch, y0 + pitch, z0 + pitch],
                        [x0, y0 + pitch, z0 + pitch],
                    ]

                    # current length of the points list, needed so the cube’s points get unique indices
                    start_idx = len(points)

                    # appends 8 points for this cube, duplicates across adjacent voxels
                    points.extend(cube)

                    # stores the 8 point indices that form this hexahedron
                    cells.append([start_idx + i for i in range(8)])

    # Remove duplicate points
    # convert the point list into a NumPy array for processing
    points = np.array(points)

    # removes duplicate coordinates
    unique_points, inverse = np.unique(points, axis=0, return_inverse=True)

    # Re-map cell corner indices from the original points index to indices in unique_points
    # Now each cell references unique node indices (0-based)
    cells = [[inverse[idx] for idx in c] for c in cells]

    print(f"🧮 {len(unique_points)} unique points, {len(cells)} voxels")

    # Write legacy ASCII VTK file manually
    with open(out_path, "w") as f:

        # Writes the VTK legacy header lines identifying file version, a comment line, data type (ASCII), and dataset type
        f.write("# vtk DataFile Version 3.0\n")
        f.write("Voxelized volume generated in Python\n")
        f.write("ASCII\n")
        f.write("DATASET UNSTRUCTURED_GRID\n")

        # Points section header with the number of points and type
        f.write(f"POINTS {len(unique_points)} float\n")

        # Writes each unique point coordinate (x y z) on its own line
        for p in unique_points:
            f.write(f"{p[0]} {p[1]} {p[2]}\n")

        # Cells section
        # For CELLS header we must give total number of integers that follow
        # for each cell we output 8 plus 8 point indices → 9 integers per cell
        total_indices = len(cells) * 9  # 8 corners + 1 count per cell

        # Writes the CELLS header: number of cells and total integers
        f.write(f"CELLS {len(cells)} {total_indices}\n")

        # For each cell, write a line beginning with 8 (number of points for this cell) 
        # followed by the 8 point indices (0-based) separated by spaces
        for c in cells:
            f.write("8 " + " ".join(map(str, c)) + "\n")

        # Cell types (12 = VTK_HEXAHEDRON)
        # Writes CELL_TYPES header with number of elements
        f.write(f"CELL_TYPES {len(cells)}\n")

        # For each cell write the cell type code 12 (VTK_HEXAHEDRON) on its own line
        # That tells readers each element is a hexahedron
        for _ in cells:
            f.write("12\n")

    print(f"✅ Saved ASCII VTK volume mesh vtk DataFile Version 3.0: {out_path}")

def save_voxel_mesh_as_msh(voxels, pitch, origin, out_path):
    """
    Save filled voxels as an ASCII .msh (Gmsh v2.2 format) with hexahedral cells.
    """
    nz, ny, nx = voxels.shape
    points = []
    cells = []

    print("🧩 Building voxel hexahedral mesh for Gmsh (.msh)...")

    # Generate all voxel cubes
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                if voxels[z, y, x]:
                    x0 = origin[0] + x * pitch
                    y0 = origin[1] + y * pitch
                    z0 = origin[2] + z * pitch
                    cube = [
                        [x0, y0, z0],
                        [x0 + pitch, y0, z0],
                        [x0 + pitch, y0 + pitch, z0],
                        [x0, y0 + pitch, z0],
                        [x0, y0, z0 + pitch],
                        [x0 + pitch, y0, z0 + pitch],
                        [x0 + pitch, y0 + pitch, z0 + pitch],
                        [x0, y0 + pitch, z0 + pitch],
                    ]
                    start_idx = len(points)
                    points.extend(cube)
                    cells.append([start_idx + i for i in range(8)])

    # Remove duplicate points
    points = np.array(points)
    unique_points, inverse = np.unique(points, axis=0, return_inverse=True)
    cells = [[inverse[idx] + 1 for idx in c] for c in cells]  # +1 for Gmsh 1-based indexing

    print(f"🧮 {len(unique_points)} unique nodes, {len(cells)} voxels")

    # Write Gmsh v2.2 ASCII format
    with open(out_path, "w") as f:

        # Write mesh format block: version 2.2, ASCII (second token=0), data size 8 bytes for double 
        # (this header is standard Gmsh v2.2)
        f.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")

        # Write node block
        f.write("$Nodes\n")

        # Write number of nodes
        f.write(f"{len(unique_points)}\n")

        # For each unique node, write a line: node_id x y z with 1-based node_id
        for i, p in enumerate(unique_points, start=1):
            f.write(f"{i} {p[0]} {p[1]} {p[2]}\n")

        # End nodes block
        f.write("$EndNodes\n")

        # Write element block
        f.write("$Elements\n")

        # Write number of elements
        f.write(f"{len(cells)}\n")

        # Set element type code: 5 is the classic Gmsh code for hexahedron (for v2.2)
        elem_type = 5  # Hexahedron

        # For each element write a line:
        # elementID elementType numTags tag1 ... node1 node2 ...
        # Here we use numTags = 1 and tag1 = 1 (a minimal tag — you can use tags to mark physical groups). 
        # Then list the 8 node IDs (1-based).
        for eid, c in enumerate(cells, start=1):
            # Format: elementID elementType numTags tag1 ... tagN node1 node2 ...
            f.write(f"{eid} {elem_type} 1 1 {' '.join(map(str, c))}\n")
        
        # Close element block
        f.write("$EndElements\n")

    print(f"✅ Saved Gmsh .msh voxel mesh: {out_path}")

def save_voxel_surface_as_stl(voxelized, out_path):
    """Save visible voxelized surface as .stl"""

    # returns a Trimesh containing cube geometry (one cube per filled voxel)
    # That mesh represents the voxelized surface geometry
    surface_mesh = voxelized.as_boxes()

    # Export that Trimesh to an STL file at out_path
    surface_mesh.export(out_path)
    print(f"✅ Saved voxelized surface as: {out_path}")

def main():
    # Create a Tk root window then hide it so only dialogs appear (no empty main window)
    root = tk.Tk()
    root.withdraw()

    # Pop up a file-open dialog for the user to pick an input mesh file
    file_path = filedialog.askopenfilename(
        title="Select a surface mesh file",
        filetypes=[("3D Mesh Files", "*.stl *.obj *.vtk")]
    )

    # If user cancels, show an info box and exit
    if not file_path:
        messagebox.showinfo("Voxelization", "No file selected. Exiting.")
        sys.exit(0)

    # Ask the user for the voxel pitch (size of a voxel in the same world units as the mesh)
    pitch = simpledialog.askfloat(
        "Voxel Size",
        "Enter voxel pitch (e.g. 0.5):",
        minvalue=0.001, maxvalue=100.0, initialvalue=0.5
    )

    # If cancelled, show info and exit
    if pitch is None:
        messagebox.showinfo("Voxelization", "No pitch provided. Exiting.")
        sys.exit(0)

    # Load the mesh
    print(f"📂 Loading mesh: {file_path}")
    mesh = load_mesh_any(file_path)

    # Warn if mesh is not watertight — voxelization and interior filling can produce holes when the input mesh has gaps
    if not mesh.is_watertight:
        print("⚠️ Warning: Mesh is not watertight — voxelization may have holes.")

    print(f"🧱 Voxelizing at pitch = {pitch} ...")

    # Create a VoxelGrid representing the surface occupancy at the given pitch
    voxelized = mesh.voxelized(pitch=pitch)

    # Fill the interior of the voxelized surface so that interior voxels are also marked (gives a solid filled volume)
    filled = voxelized.fill()

    # filled.matrix is a 3D boolean/uint array (shape z,y,x) with True for filled voxels
    voxels = filled.matrix

    # Extract the world-space origin (corner) of the voxel grid from the transform matrix of the VoxelGrid
    # This associates voxel indices with world coordinates
    origin = filled.transform[:3, 3]

    print(f"Voxel grid shape: {voxels.shape}")

    # Compute output directory and base filename, and construct full paths for the output files
    out_dir = os.path.dirname(file_path)
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    vtk_out = os.path.join(out_dir, f"{base_name}_voxel_volume_mesh.vtk")
    stl_out = os.path.join(out_dir, f"{base_name}_voxel_surface.stl")
    msh_out = os.path.join(out_dir, f"{base_name}_voxel_volume.msh")

    # Write the ASCII legacy .vtk file that contains unique points and hexahedral cells
    save_voxel_mesh_as_vtk(voxels, pitch, origin, vtk_out)

    # Export the voxel surface as an STL of cubes
    save_voxel_surface_as_stl(filled, stl_out)
    
    # Write a Gmsh v2.2 ASCII .msh file that contains nodes and hexahedral element
    save_voxel_mesh_as_msh(voxels, pitch, origin, msh_out)

    # Pop up a message box informing the user that voxelization is finished and listing saved files
    messagebox.showinfo(
        "Voxelization Complete",
        f"Voxelization complete!\n\nSaved:\n{vtk_out}\n{stl_out}"
    )

if __name__ == "__main__":
    main()
