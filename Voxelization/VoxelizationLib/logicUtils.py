import vtk
import vtk.util.numpy_support as vtk_np

def exportModelVTK(modelNode, filePath):
    """Writes the polydata to a VTK (.vtk) file."""
    if not modelNode:
        return
    
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(filePath)
    writer.SetInputData(modelNode.GetPolyData())
    writer.Write()

def exportModelSTL(modelNode, filePath):
    """Writes the polydata to an STL (.stl) file."""
    if not modelNode:
        return
        
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
    
    
def getVoxelizedModel(inputModel, pitch, outputModel):
    import trimesh
    from numpy import hstack, full, int64
    
    inputModel.HardenTransform()    
    polyData = inputModel.GetPolyData()

    points = vtk.util.numpy_support.vtk_to_numpy(polyData.GetPoints().GetData())
    cells = vtk.util.numpy_support.vtk_to_numpy(polyData.GetPolys().GetData())
    faces = cells.reshape(-1, 4)[:, 1:]

    mesh = trimesh.Trimesh(vertices=points, faces=faces, process=False)
    voxelized = mesh.voxelized(pitch=pitch).fill()

    surface_mesh = voxelized.as_boxes()
    v_out = surface_mesh.vertices
    f_out = surface_mesh.faces

    out_poly = vtk.vtkPolyData()

    v_vtk = vtk.util.numpy_support.numpy_to_vtk(v_out, deep=True)
    pts = vtk.vtkPoints()
    pts.SetData(v_vtk)
    out_poly.SetPoints(pts)

    num_faces = f_out.shape[0]
    cells_array = hstack([full((num_faces, 1), 3), f_out]).astype(int64)
    cells_vtk = vtk.util.numpy_support.numpy_to_vtkIdTypeArray(cells_array, deep=True)

    connectivity = vtk.vtkCellArray()
    connectivity.SetCells(num_faces, cells_vtk)
    out_poly.SetPolys(connectivity)

    outputModel.SetAndObservePolyData(out_poly)
    
    return outputModel
    
def displayVoxelizedModel(voxelizedModel) -> None:
    if not voxelizedModel.GetDisplayNode():
            voxelizedModel.CreateDefaultDisplayNodes()

    voxelizedModel.GetDisplayNode().SetVisibility(True)
    voxelizedModel.GetPolyData().Modified()
    
def computeMetrics(grid_original, grid_voxelized, originalVoxelCount, voxelizedVoxelCount) -> dict:
    from numpy import sum, logical_and, logical_or
    
    intersection = logical_and(grid_original, grid_voxelized)
    union = logical_or(grid_original, grid_voxelized)

    intersection_count = sum(intersection)
    union_count = sum(union)

    # Dice
    dice = (2.0 * intersection_count) / (originalVoxelCount + voxelizedVoxelCount) if (originalVoxelCount + voxelizedVoxelCount) > 0 else 0.0

    # IoU
    iou = intersection_count / union_count if union_count > 0 else 0.0

    # DeltaV
    deltaV = (abs(voxelizedVoxelCount - originalVoxelCount) / voxelizedVoxelCount) * 100 if originalVoxelCount > 0 else 0.0

    return {
        "dice": dice,
        "iou": iou,
        "deltaV": deltaV 
        }