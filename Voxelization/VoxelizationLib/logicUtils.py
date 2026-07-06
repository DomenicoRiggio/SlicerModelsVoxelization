import vtk
import vtk.util.numpy_support as vtk_np

def exportModelOBJ(modelNode, filePath):
    """
    Writes the polydata to a Wavefront OBJ file compatible with Gmsh 4.x.
    Uses plain ASCII with no comment lines, no headers, no mtl references.
    """
    if not modelNode or not modelNode.GetPolyData() or modelNode.GetPolyData().GetNumberOfPoints() == 0:
        raise ValueError("Output model has no geometry. Run voxelization first.")

    # Triangulate
    triangulate = vtk.vtkTriangleFilter()
    triangulate.SetInputData(modelNode.GetPolyData())
    triangulate.Update()
    polyData = triangulate.GetOutput()

    points     = vtk.util.numpy_support.vtk_to_numpy(polyData.GetPoints().GetData())
    n_cells    = polyData.GetPolys().GetNumberOfCells()
    cell_array = vtk.util.numpy_support.vtk_to_numpy(polyData.GetPolys().GetData())
    triangles  = cell_array.reshape(n_cells, 4)[:, 1:]

    lines = []
    for p in points:
        lines.append("v " + str(float(p[0])) + " " + str(float(p[1])) + " " + str(float(p[2])))
    for t in triangles:
        lines.append("f " + str(int(t[0])+1) + " " + str(int(t[1])+1) + " " + str(int(t[2])+1))

    with open(filePath, 'w', newline='\n') as f:
        f.write('\n'.join(lines) + '\n')

    print(f"OBJ saved to {filePath}")


def exportModelPLY(modelNode, filePath):
    """
    Writes the polydata to a PLY file.
    PLY is widely supported by MeshLab, CloudCompare, Blender, open3d.
    """
    if not modelNode or not modelNode.GetPolyData() or modelNode.GetPolyData().GetNumberOfPoints() == 0:
        raise ValueError("Output model has no geometry. Run voxelization first.")

    writer = vtk.vtkPLYWriter()
    writer.SetFileName(filePath)
    writer.SetInputData(modelNode.GetPolyData())
    writer.SetFileTypeToBinary()
    writer.Write()
    print(f"PLY saved to {filePath}")


def exportModelOFF(modelNode, filePath):
    """
    Writes the polydata to an OFF (Object File Format) file.
    OFF is supported by MeshLab, many geometry processing research tools.
    Written manually — no extra dependency needed.
    """
    if not modelNode or not modelNode.GetPolyData() or modelNode.GetPolyData().GetNumberOfPoints() == 0:
        raise ValueError("Output model has no geometry. Run voxelization first.")

    triangulate = vtk.vtkTriangleFilter()
    triangulate.SetInputData(modelNode.GetPolyData())
    triangulate.Update()
    polyData = triangulate.GetOutput()

    points     = vtk.util.numpy_support.vtk_to_numpy(polyData.GetPoints().GetData())
    n_cells    = polyData.GetPolys().GetNumberOfCells()
    cell_array = vtk.util.numpy_support.vtk_to_numpy(polyData.GetPolys().GetData())
    triangles  = cell_array.reshape(n_cells, 4)[:, 1:]

    with open(filePath, 'w', newline='\n') as f:
        f.write("OFF\n")
        f.write(f"{len(points)} {n_cells} 0\n")
        for p in points:
            f.write(str(float(p[0])) + " " + str(float(p[1])) + " " + str(float(p[2])) + "\n")
        for t in triangles:
            f.write("3 " + str(int(t[0])) + " " + str(int(t[1])) + " " + str(int(t[2])) + "\n")

    print(f"OFF saved to {filePath}")


def exportModelVTK(modelNode, filePath):
    """Writes the polydata to a VTK (.vtk) file."""
    if not modelNode or not modelNode.GetPolyData() or modelNode.GetPolyData().GetNumberOfPoints() == 0:
        raise ValueError("Output model has no geometry. Run voxelization first.")

    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(filePath)
    writer.SetInputData(modelNode.GetPolyData())
    writer.Write()

def exportModelSTL(modelNode, filePath):
    """Writes the polydata to an STL (.stl) file."""
    if not modelNode or not modelNode.GetPolyData() or modelNode.GetPolyData().GetNumberOfPoints() == 0:
        raise ValueError("Output model has no geometry. Run voxelization first.")
        
    writer = vtk.vtkSTLWriter()
    writer.SetFileName(filePath)
    writer.SetInputData(modelNode.GetPolyData())

    # STL files usually require binary format for smaller file size
    writer.SetFileTypeToBinary()
    writer.Write()

def exportModelMSH(modelNode, filePath):
    import meshio
    """Writes the polydata to a Gmsh (.msh) file using meshio."""
    # Ensure the mesh is purely triangles
    triangulate = vtk.vtkTriangleFilter()
    triangulate.SetInputData(modelNode.GetPolyData())
    triangulate.Update()
    polyData = triangulate.GetOutput()
    
    # Extract points / nodes
    points = vtk_np.vtk_to_numpy(polyData.GetPoints().GetData())

    # Extract cells / elements
    vtk_cells = polyData.GetPolys()
    n_cells = vtk_cells.GetNumberOfCells()
    cell_array = vtk_np.vtk_to_numpy(vtk_cells.GetData())
    
    # Reshape for triangles: skips the '3' count element at index 0, 4, 8...
    triangles = cell_array.reshape(n_cells, 4)[:, 1:]

    # Write using meshio forced to version 2.2 ASCII
    cells = [("triangle", triangles)]
    mesh = meshio.Mesh(points, cells)
    
    # Specify gmsh22 to ensure maximum compatibility with the Gmsh parser
    mesh.write(filePath, file_format="gmsh22", binary=False)
    print(f"MSH (Version 2.2 ASCII) saved to {filePath}")


def rasterizeModelToVolume(modelNode, referenceVolume):
    import numpy as np

    modelNode.HardenTransform()


    # RAS to IJK matrix
    rasToIjk = vtk.vtkMatrix4x4()
    referenceVolume.GetRASToIJKMatrix(rasToIjk)

    transform = vtk.vtkTransform()
    transform.SetMatrix(rasToIjk)

    tfFilter = vtk.vtkTransformPolyDataFilter()
    tfFilter.SetInputData(modelNode.GetPolyData())
    tfFilter.SetTransform(transform)
    tfFilter.Update()

    poly_ijk = tfFilter.GetOutput()


    img = referenceVolume.GetImageData()
    extent = img.GetExtent()
    dims = (
        extent[1]-extent[0]+1,
        extent[3]-extent[2]+1,
        extent[5]-extent[4]+1
    )

    # create empty mask
    mask = np.zeros((dims[2], dims[1], dims[0]), dtype=np.uint8)

    whiteImage = vtk.vtkImageData()
    whiteImage.SetDimensions(dims)
    whiteImage.SetExtent(0, dims[0]-1, 0, dims[1]-1, 0, dims[2]-1)
    whiteImage.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
    whiteImage.GetPointData().GetScalars().Fill(1)

    polyToStencil = vtk.vtkPolyDataToImageStencil()
    polyToStencil.SetInputData(poly_ijk)
    polyToStencil.SetOutputWholeExtent(whiteImage.GetExtent())
    polyToStencil.Update()

    imgStencil = vtk.vtkImageStencil()
    imgStencil.SetInputData(whiteImage)
    imgStencil.SetStencilConnection(polyToStencil.GetOutputPort())
    imgStencil.SetBackgroundValue(0)
    imgStencil.Update()

    outImg = imgStencil.GetOutput()

    arr = vtk.util.numpy_support.vtk_to_numpy(outImg.GetPointData().GetScalars())
    arr = arr.reshape(mask.shape)

    return arr
    
    
def getVoxelizedModel(inputModel, pitch, outputModel, threshold=0.0):
    """
    Voxelize inputModel at the given pitch and apply an occupancy threshold.

    Occupancy strategy (no ray casting, no rtree required)
    -------------------------------------------------------
    After building the filled voxel grid we classify each voxel as:

    - Interior voxel  : present in the FILLED grid but NOT in the
                        SURFACE-ONLY grid  →  occupancy = 1.0  (always kept)
    - Surface voxel   : present in BOTH grids  →  occupancy estimated by
                        checking how many of the voxel's 8 corner points
                        lie inside the mesh via signed-distance query.
                        occupancy = corners_inside / 8

    This avoids ray casting entirely so no rtree dependency is needed.

    threshold values
    ----------------
    0.0  keep every voxel touched by the mesh (fast path, original behaviour)
    0.5  keep surface voxels at least half-inside the mesh
    1.0  keep only fully interior voxels

    Parameters
    ----------
    inputModel  : vtkMRMLModelNode
    pitch       : float  - voxel side length in mm
    outputModel : vtkMRMLModelNode
    threshold   : float [0.0, 1.0]
    """
    import trimesh
    import numpy as np
    from numpy import hstack, full, int64

    inputModel.HardenTransform()
    polyData = inputModel.GetPolyData()

    # Convert VTK polydata to trimesh
    points = vtk.util.numpy_support.vtk_to_numpy(polyData.GetPoints().GetData())
    cells  = vtk.util.numpy_support.vtk_to_numpy(polyData.GetPolys().GetData())
    faces  = cells.reshape(-1, 4)[:, 1:]
    mesh   = trimesh.Trimesh(vertices=points, faces=faces, process=False)

    if threshold <= 0.0:
        # ----------------------------------------------------------------
        # Fast path — no filtering, original behaviour
        # No voxels are excluded so excludedCenters is empty.
        # ----------------------------------------------------------------
        voxelized         = mesh.voxelized(pitch=pitch).fill()
        surface_mesh      = voxelized.as_boxes()
        excludedCenters   = np.empty((0, 3), dtype=float)
        excludedOccupancy = np.empty((0,), dtype=float)

    else:
        # ----------------------------------------------------------------
        # Occupancy path
        # For each filled voxel we place a 4x4x4 grid of 64 sample points
        # INSIDE that voxel and test how many are inside the mesh using
        # vtkSelectEnclosedPoints (no rtree, no trimesh proximity needed).
        #
        #   occupancy = points_inside_mesh / 64
        #
        # Voxels with occupancy >= threshold are kept.
        # ----------------------------------------------------------------
        import numpy as np

        # Step 1 — build the filled voxel grid
        voxelGrid    = mesh.voxelized(pitch=pitch).fill()
        voxelCenters = voxelGrid.points          # (N, 3) world coords of voxel centers
        denseMatrix  = voxelGrid.matrix.copy()   # (X, Y, Z) bool — will be filtered
        N            = len(voxelCenters)

        # Step 2 — build a 4x4x4 local sample grid inside one unit voxel.
        # Points are placed at the centres of 64 equal sub-cells so they
        # are evenly distributed and never on the voxel boundary.
        n       = 4
        subSize = pitch / n
        offsets = np.linspace(-pitch/2 + subSize/2, pitch/2 - subSize/2, n)
        gx, gy, gz = np.meshgrid(offsets, offsets, offsets, indexing='ij')
        localGrid  = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
        # localGrid shape: (64, 3)

        # Step 3 — generate ALL sample points for ALL voxels in one array
        # shape: (N*64, 3)
        allSamples = (
            voxelCenters[:, np.newaxis, :]   # (N, 1, 3)
            + localGrid[np.newaxis, :, :]    # (1, 64, 3)
        ).reshape(-1, 3)                     # (N*64, 3)

        # Step 4 — build VTK polydata for the mesh surface (used by enclosure test)
        meshPolyData = vtk.vtkPolyData()
        meshPts      = vtk.vtkPoints()
        meshPts.SetData(vtk.util.numpy_support.numpy_to_vtk(mesh.vertices, deep=True))
        meshPolyData.SetPoints(meshPts)
        meshFacesArr = np.hstack([
            np.full((len(mesh.faces), 1), 3, dtype=np.int64),
            mesh.faces.astype(np.int64)
        ])
        meshCells = vtk.vtkCellArray()
        meshCells.SetCells(
            len(mesh.faces),
            vtk.util.numpy_support.numpy_to_vtkIdTypeArray(meshFacesArr.ravel(), deep=True)
        )
        meshPolyData.SetPolys(meshCells)

        # Step 5 — build VTK polydata for the sample points
        samplePolyData = vtk.vtkPolyData()
        samplePts      = vtk.vtkPoints()
        samplePts.SetData(vtk.util.numpy_support.numpy_to_vtk(allSamples, deep=True))
        samplePolyData.SetPoints(samplePts)

        # Step 6 — run vtkSelectEnclosedPoints: outputs 1=inside, 0=outside
        selectEnclosed = vtk.vtkSelectEnclosedPoints()
        selectEnclosed.SetInputData(samplePolyData)
        selectEnclosed.SetSurfaceData(meshPolyData)
        selectEnclosed.SetTolerance(0.0001)
        selectEnclosed.Update()

        insideArray = vtk.util.numpy_support.vtk_to_numpy(
            selectEnclosed.GetOutput().GetPointData().GetArray("SelectedPoints")
        ).astype(bool)                       # shape (N*64,)

        # Step 7 — compute occupancy per voxel
        # reshape to (N, 64), sum along axis=1, divide by 64
        occupancy = insideArray.reshape(N, n**3).sum(axis=1) / float(n**3)
        # occupancy[i] = fraction of 64 points inside the mesh for voxel i

        # Step 8 — discard voxels below threshold in the dense matrix
        # and collect their world-space centers and occupancy values
        filledIndices    = np.argwhere(denseMatrix)  # same order as voxelGrid.points
        excludedMask     = occupancy < threshold
        excludedCenters  = voxelCenters[excludedMask]
        excludedOccupancy = occupancy[excludedMask] * 100.0  # convert to percentage

        for fi, occ in zip(filledIndices, occupancy):
            if occ < threshold:
                denseMatrix[fi[0], fi[1], fi[2]] = False

        # Step 9 — rebuild filtered VoxelGrid and convert to surface mesh
        filteredGrid = trimesh.voxel.VoxelGrid(
            trimesh.voxel.encoding.DenseEncoding(denseMatrix),
            voxelGrid.transform
        )
        surface_mesh = filteredGrid.as_boxes()

    # ------------------------------------------------------------------
    # Convert trimesh surface back to VTK polydata
    # ------------------------------------------------------------------
    v_out = surface_mesh.vertices
    f_out = surface_mesh.faces

    out_poly = vtk.vtkPolyData()

    v_vtk = vtk.util.numpy_support.numpy_to_vtk(v_out, deep=True)
    pts   = vtk.vtkPoints()
    pts.SetData(v_vtk)
    out_poly.SetPoints(pts)

    num_faces   = f_out.shape[0]
    cells_array = hstack([full((num_faces, 1), 3), f_out]).astype(int64)
    cells_vtk   = vtk.util.numpy_support.numpy_to_vtkIdTypeArray(cells_array, deep=True)

    connectivity = vtk.vtkCellArray()
    connectivity.SetCells(num_faces, cells_vtk)
    out_poly.SetPolys(connectivity)

    outputModel.SetAndObservePolyData(out_poly)

    return outputModel, excludedOccupancy


def _emptyStats():
    return {"mean": 0.0, "std": 0.0, "median": 0.0, "p5": 0.0, "p95": 0.0}
    
def displayVoxelizedModel(voxelizedModel) -> None:
    if not voxelizedModel.GetDisplayNode():
        voxelizedModel.CreateDefaultDisplayNodes()
    dn = voxelizedModel.GetDisplayNode()
    dn.SetVisibility(True)
    dn.SetOpacity(1.0)
    dn.SetRepresentation(dn.SurfaceRepresentation)
    voxelizedModel.SetAttribute("Terminologies.TerminologyEntry", "")
    voxelizedModel.GetPolyData().Modified()
    
def computeVolumeCm3(modelNode) -> float:
    """
    Compute the volume of a closed surface mesh in cm³ using VTK's
    vtkMassProperties filter.

    vtkMassProperties works on any closed triangulated surface and returns
    the signed volume directly — no voxel count or pitch needed.
    Result is converted from mm³ (Slicer's internal unit) to cm³.
    """
    # Triangulate first — vtkMassProperties requires pure triangles
    triangulate = vtk.vtkTriangleFilter()
    triangulate.SetInputData(modelNode.GetPolyData())
    triangulate.Update()

    massProps = vtk.vtkMassProperties()
    massProps.SetInputData(triangulate.GetOutput())
    massProps.Update()

    volumeMm3 = massProps.GetVolume()   # in mm³
    return abs(volumeMm3) / 1000.0      # convert to cm³


def computeMetrics(grid_original, grid_voxelized, originalVoxelCount, voxelizedVoxelCount, originalVolCm3=0.0, voxelizedVolCm3=0.0) -> dict:
    from numpy import sum

    # DeltaV percentage — deltaV(cm³) / voxelized volume × 100
    deltaVCm3 = abs(voxelizedVolCm3 - originalVolCm3)
    deltaVPct = (deltaVCm3 / voxelizedVolCm3) * 100 if voxelizedVolCm3 > 0 else 0.0

    return {
        "deltaV":    deltaVPct,
        "deltaVCm3": deltaVCm3,
    }


def computeIntensityStats(excludedOccupancy) -> dict:
    """
    Compute statistics on the occupancy percentages of excluded voxels.

    Each excluded voxel has an occupancy value between 0% and threshold*100%.
    For example if threshold=0.5, all values here are between 0% and 50%.

    This tells the user:
    - Mean/std: on average how filled were the discarded voxels
    - Median/p5/p95: distribution of fill levels in discarded voxels

    A mean close to 0% means mostly empty voxels were discarded (good).
    A mean close to threshold*100% means many borderline voxels were
    discarded — consider lowering the threshold.

    Parameters
    ----------
    excludedOccupancy : np.ndarray shape (M,)
        Occupancy percentage (0-100) of each excluded voxel.
    """
    import numpy as np

    empty = {"mean": 0.0, "std": 0.0, "median": 0.0, "p5": 0.0, "p95": 0.0}

    if excludedOccupancy is None or len(excludedOccupancy) == 0:
        return empty

    return {
        "mean":   float(np.mean(excludedOccupancy)),
        "std":    float(np.std(excludedOccupancy)),
        "median": float(np.median(excludedOccupancy)),
        "p5":     float(np.percentile(excludedOccupancy, 5)),
        "p95":    float(np.percentile(excludedOccupancy, 95)),
    }