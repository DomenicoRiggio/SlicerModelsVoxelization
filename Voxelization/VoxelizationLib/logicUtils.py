import vtk
import vtk.util.numpy_support as vtk_np

def exportModelOBJ(modelNode, filePath):
    """Export model as OBJ using Slicer's built-in export with AddDefaultStorageNode."""
    import slicer
    if not modelNode or not modelNode.GetPolyData() or modelNode.GetPolyData().GetNumberOfPoints() == 0:
        raise ValueError("Output model has no geometry. Run voxelization first.")
    modelNode.AddDefaultStorageNode()
    slicer.util.saveNode(modelNode, filePath)


def exportModelVTK(modelNode, filePath):
    """Export model as VTK using Slicer's built-in export with AddDefaultStorageNode."""
    import slicer
    if not modelNode or not modelNode.GetPolyData() or modelNode.GetPolyData().GetNumberOfPoints() == 0:
        raise ValueError("Output model has no geometry. Run voxelization first.")
    modelNode.AddDefaultStorageNode()
    slicer.util.saveNode(modelNode, filePath)


def exportModelSTL(modelNode, filePath):
    """Export model as STL using Slicer's built-in export with AddDefaultStorageNode."""
    import slicer
    if not modelNode or not modelNode.GetPolyData() or modelNode.GetPolyData().GetNumberOfPoints() == 0:
        raise ValueError("Output model has no geometry. Run voxelization first.")
    modelNode.AddDefaultStorageNode()
    slicer.util.saveNode(modelNode, filePath)


def exportModelPLY(modelNode, filePath):
    """Export model as PLY using Slicer's native PLY support via AddDefaultStorageNode."""
    import slicer
    if not modelNode or not modelNode.GetPolyData() or modelNode.GetPolyData().GetNumberOfPoints() == 0:
        raise ValueError("Output model has no geometry. Run voxelization first.")
    modelNode.AddDefaultStorageNode()
    slicer.util.saveNode(modelNode, filePath)


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
    Voxelize inputModel at the given pitch using trimesh.

    trimesh.voxelized(pitch).fill() correctly captures ALL voxels touching
    the surface including boundary voxels. VTK stencil alternatives only
    capture strictly interior voxels, missing surface-touching voxels.

    threshold=0 : fast path
    threshold>0 : 64-point occupancy filtering per voxel using vtkSelectEnclosedPoints
    """
    import trimesh
    import numpy as np
    from numpy import hstack, full, int64

    inputModel.HardenTransform()
    polyData = inputModel.GetPolyData()

    points = vtk.util.numpy_support.vtk_to_numpy(polyData.GetPoints().GetData())
    cells  = vtk.util.numpy_support.vtk_to_numpy(polyData.GetPolys().GetData())
    faces  = cells.reshape(-1, 4)[:, 1:]
    mesh   = trimesh.Trimesh(vertices=points, faces=faces, process=False)

    if threshold <= 0.0:
        voxelized         = mesh.voxelized(pitch=pitch).fill()
        surface_mesh      = voxelized.as_boxes()
        excludedOccupancy = np.empty((0,), dtype=float)
    else:
        voxelGrid    = mesh.voxelized(pitch=pitch).fill()
        voxelCenters = voxelGrid.points
        denseMatrix  = voxelGrid.matrix.copy()
        N            = len(voxelCenters)

        n       = 4
        subSize = pitch / n
        offsets = np.linspace(-pitch/2 + subSize/2, pitch/2 - subSize/2, n)
        gx, gy, gz = np.meshgrid(offsets, offsets, offsets, indexing="ij")
        localGrid  = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])

        allSamples = (
            voxelCenters[:, np.newaxis, :] + localGrid[np.newaxis, :, :]
        ).reshape(-1, 3)

        samplePolyData = vtk.vtkPolyData()
        samplePts      = vtk.vtkPoints()
        samplePts.SetData(vtk.util.numpy_support.numpy_to_vtk(allSamples, deep=True))
        samplePolyData.SetPoints(samplePts)

        selectEnclosed = vtk.vtkSelectEnclosedPoints()
        selectEnclosed.SetInputData(samplePolyData)
        selectEnclosed.SetSurfaceData(polyData)
        selectEnclosed.SetTolerance(0.0001)
        selectEnclosed.Update()

        insideArray = vtk.util.numpy_support.vtk_to_numpy(
            selectEnclosed.GetOutput().GetPointData().GetArray("SelectedPoints")
        ).astype(bool)

        occupancy    = insideArray.reshape(N, n**3).sum(axis=1) / float(n**3)
        excludedMask = occupancy < threshold
        excludedOccupancy = occupancy[excludedMask] * 100.0

        filledIndices = np.argwhere(denseMatrix)
        for fi, occ in zip(filledIndices, occupancy):
            if occ < threshold:
                denseMatrix[fi[0], fi[1], fi[2]] = False

        filteredGrid = trimesh.voxel.VoxelGrid(
            trimesh.voxel.encoding.DenseEncoding(denseMatrix),
            voxelGrid.transform
        )
        surface_mesh = filteredGrid.as_boxes()

    v_out = surface_mesh.vertices
    f_out = surface_mesh.faces

    out_poly = vtk.vtkPolyData()
    v_vtk    = vtk.util.numpy_support.numpy_to_vtk(v_out, deep=True)
    pts      = vtk.vtkPoints()
    pts.SetData(v_vtk)
    out_poly.SetPoints(pts)

    num_faces   = f_out.shape[0]
    cells_array = hstack([full((num_faces, 1), 3), f_out]).astype(int64)
    cells_vtk   = vtk.util.numpy_support.numpy_to_vtkIdTypeArray(
        cells_array.ravel(), deep=True
    )
    connectivity = vtk.vtkCellArray()
    connectivity.SetCells(num_faces, cells_vtk)
    out_poly.SetPolys(connectivity)

    outputModel.SetAndObservePolyData(out_poly)
    return outputModel, excludedOccupancy

def _emptyStats():
    return {"mean": 0.0, "std": 0.0, "median": 0.0, "p5": 0.0, "p95": 0.0}
    
def displayVoxelizedModel(voxelizedModel) -> None:
    """
    Ensure the model has a display node and is visible.
    Uses Slicer's CreateDefaultDisplayNodes() infrastructure.
    """
    import slicer
    if not voxelizedModel:
        return
    # Use Slicer's standard infrastructure to create display nodes
    if not voxelizedModel.GetDisplayNode():
        voxelizedModel.CreateDefaultDisplayNodes()
    dn = voxelizedModel.GetDisplayNode()
    if dn is None:
        return
    dn.SetVisibility(True)
    dn.SetOpacity(1.0)
    dn.SetRepresentation(dn.SurfaceRepresentation)
    # Suppress terminology warnings using Slicer's attribute system
    voxelizedModel.SetAttribute("Terminologies.TerminologyEntry", "")
    voxelizedModel.GetPolyData().Modified()
    
def computeVolumeCm3(modelNode) -> float:
    """
    Compute the volume of a closed surface mesh in cm³.
    Uses vtkMassProperties — the same approach used internally by
    Slicer's Segment Statistics module for surface mesh volume.
    Result is converted from mm³ to cm³.
    """
    triangulate = vtk.vtkTriangleFilter()
    triangulate.SetInputData(modelNode.GetPolyData())
    triangulate.Update()

    massProps = vtk.vtkMassProperties()
    massProps.SetInputData(triangulate.GetOutput())
    massProps.Update()

    volumeMm3 = massProps.GetVolume()
    return abs(volumeMm3) / 1000.0


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