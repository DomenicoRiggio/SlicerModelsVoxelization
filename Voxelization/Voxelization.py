import logging
import os
from typing import Annotated, Optional

import vtk
import vtk.util.numpy_support as vtk_np
import numpy as np

import slicer
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import (
    parameterNodeWrapper,
    WithinRange,
)

from slicer import vtkMRMLModelNode


#
# Voxelization
#


class Voxelization(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("Voxelization")
        # TODO: set categories (folders where the module shows up in the module selector)
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "Examples")]
        self.parent.dependencies = []  # TODO: add here list of module names that this module requires
        self.parent.contributors = ["Laura Lichtlein (KIT Institute of Biomedical Engineering)"]
        self.parent.helpText = _("""
This module provides tools for 3D model manipulation and export. Features include uniformly resizing models, 
converting surface meshes into solid cubical voxel models and exporting processed the models to a chosen 
directory in .vtk, .stl, and .msh (Gmsh 2.2 ASCII) formats.
See more information in <a href="https://github.com/organization/projectname#Voxelization">module documentation</a>.
""")
        # TODO: replace with organization, grant and thanks
        self.parent.acknowledgementText = _("""
This file was originally developed by Laura Lichtlein, KIT Institute of Biomedical Engingeering.
""")




#
# VoxelizationParameterNode
#


@parameterNodeWrapper
class VoxelizationParameterNode:
    """
    The parameters needed by module.

    inputModel  - The VTK model to be voxelized.
    pitch       - The value setting the side length for the voxels.
    inputModel  - The VTK model to be exported.
    """

    inputModel: vtkMRMLModelNode
    pitch: float
    outputModel: vtkMRMLModelNode


#
# VoxelizationWidget
#


class VoxelizationWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self._parameterNode = None
        self._parameterNodeGuiTag = None
        self.requiredDeps = ["trimesh", "meshio"]
        
    def checkDependencies(self):
        from importlib.util import find_spec
        allPresent = all(find_spec(mod) is not None for mod in self.requiredDeps)
        
        if not allPresent:
            if not slicer.util.confirmOkCancelDisplay(
                        "The dependencies needed for the extension will be installed, the operation may take a few minutes. A Slicer restart will be necessary.",
                        "Press OK to install and restart."
                    ):
                        raise ValueError("Missing dependencies.")

            
            slicer.util.setPythonConsoleVisible(True)
            print(f"Installing missing dependencies, please wait...")

            try:
                for dep in self.requiredDeps:
                    print(f"Installing {dep}...")
                    slicer.util.pip_install(dep)
                                
                print(f"All dependencies installed successfully.")
                slicer.app.restart()
            except Exception as e:
                slicer.util.errorDisplay(f"Failed to install requirements: {e}")            
        

    def setup(self) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/Voxelization.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class that implements all computations.
        self.logic = VoxelizationLogic()

        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        # Buttons
        #self.ui.scaleButton.connect("clicked(bool)", self.onScaleButton)
        self.ui.voxelButton.connect("clicked(bool)", self.onVoxelButton)
        self.ui.exportToFileButton.connect("clicked(bool)", self.onExportButton)

        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()
        self.checkDependencies()

    def cleanup(self) -> None:
        """Called when the application closes and the module widget is destroyed."""
        self.removeObservers()

    def enter(self) -> None:
        """Called each time the user opens this module."""
        # Make sure parameter node exists and observed
        self.initializeParameterNode()

    def exit(self) -> None:
        """Called each time the user opens a different module."""
        # Do not react to parameter node changes (GUI will be updated when the user enters into the module)
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self._parameterNodeGuiTag = None
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)

    def onSceneStartClose(self, caller, event) -> None:
        """Called just before the scene is closed."""
        # Parameter node will be reset, do not use it anymore
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event) -> None:
        """Called just after the scene is closed."""
        # If this module is shown while the scene is closed then recreate a new parameter node immediately
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self) -> None:
        """Ensure parameter node exists and observed."""
        # Parameter node stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.

        self.setParameterNode(self.logic.getParameterNode())

        # Select default input nodes if nothing is selected yet to save a few clicks for the user
        if self._parameterNode and not self._parameterNode.inputModel:
            firstModelNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLModelNode")
            if firstModelNode:
                self._parameterNode.inputModel = firstModelNode

    def setParameterNode(self, inputParameterNode: Optional[VoxelizationParameterNode]) -> None:
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """

        """ if self._parameterNode:
                self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
                self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)
        """
        self._parameterNode = inputParameterNode
        if self._parameterNode:
            # Note: in the .ui file, a Qt dynamic property called "SlicerParameterName" is set on each
            # ui element that needs connection.
            self._parameterNodeGuiTag = self._parameterNode.connectGui(self.ui)
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)
            self._checkCanApply()

    def _checkCanApply(self, caller=None, event=None) -> None:
        if self._parameterNode and self._parameterNode.inputModel and self._parameterNode.outputModel:
            self.ui.exportToFileButton.enabled = True
            self.ui.exportToFileButton.toolTip = _("Export model")

            self.ui.voxelButton.enabled = True
            self.ui.voxelButton.toolTip = _("Voxelize model")
        else:
            self.ui.exportToFileButton.enabled = False
            self.ui.exportToFileButton.toolTip = _("Select input and output models")

            self.ui.voxelButton.enabled = False
            self.ui.voxelButton.toolTip = _("Select input and output models")

    """def onScaleButton(self) -> None:
        #Scale the input model in size

        with slicer.util.tryWithErrorDisplay(_("Decimation failed."), waitCursor=True):
            inputModel = self.ui.inputSelector.currentNode()
            outputModel = self.ui.outputSelectorModel.currentNode()
            
            # Map the slider value to the scaling factor
            scaleFactor = float(self.ui.scaleWidget.value)
            
            # If your output selector is empty, we can overwrite the input
            if not outputModel:
                outputModel = inputModel

            with slicer.util.tryWithErrorDisplay(_("Operation failed."), waitCursor=True):
                self.logic.scaleModel(inputModel, outputModel, scaleFactor)"""

    def onVoxelButton(self) -> None:
        """Voxelize the input model"""

        with slicer.util.tryWithErrorDisplay(_("Voxelization failed."), waitCursor=True):
            inputModel = self.ui.inputSelector.currentNode()
            outputModel = self.ui.outputSelectorModel.currentNode()
            
            # Map the slider value to the pitch
            pitch = float(self.ui.pitchWidget.value)

            # If your output selector is empty, we can overwrite the input
            if not outputModel:
                outputModel = inputModel

            with slicer.util.tryWithErrorDisplay(_("Operation failed."), waitCursor=True):
                #self.logic.voxelizeModelToModel(inputModel, outputModel, pitch, self.ui)
                self.logic.voxelizeModelAndCompare(inputModel, outputModel, pitch, self.ui)
                
    def onExportButton(self) -> None:
        """Export output model to file"""
        outputModel = self.ui.outputSelectorModel.currentNode()
        directory = self.ui.DirectoryButton.directory

        if not outputModel:
            slicer.util.errorDisplay("Please select an Output Model first.")
            return
        if not directory:
            slicer.util.errorDisplay("Please select a save directory.")
            return

        # Define file paths for saving the model as .vtk .stl .msh
        baseFileName = outputModel.GetName()
        vtkPath = os.path.join(directory, f"{baseFileName}.vtk")
        stlPath = os.path.join(directory, f"{baseFileName}.stl")
        mshPath = os.path.join(directory, f"{baseFileName}.msh")

        with slicer.util.tryWithErrorDisplay(_("Export failed."), waitCursor=True):
            # Export VTK
            self.logic.exportModelVTK(outputModel, vtkPath)
            # Export STL
            self.logic.exportModelSTL(outputModel, stlPath)
            # Export MSH
            self.logic.exportModelMSH(outputModel, mshPath)

            slicer.util.delayDisplay(f"Model saved to:\n{directory}")


#
# VoxelizationLogic
#


class VoxelizationLogic(ScriptedLoadableModuleLogic):
    """This class implements all the actual
    computation done by the module: Scaling the model, voxelizing the model and exporting it to files.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self) -> None:
        """Called when the logic class is instantiated. Can be used for initializing member variables."""
        ScriptedLoadableModuleLogic.__init__(self)

    def getParameterNode(self):
        return VoxelizationParameterNode(super().getParameterNode())

    def computeVoxelMetrics(self, maskA: np.ndarray, maskB: np.ndarray) -> dict:
        if maskA.shape != maskB.shape:
            raise ValueError("Voxel grids must have the same shape")

        intersection = np.logical_and(maskA, maskB).sum()
        union = np.logical_or(maskA, maskB).sum()

        volA = maskA.sum()
        volB = maskB.sum()

        dice = 2.0 * intersection / (volA + volB) if (volA + volB) > 0 else 1.0
        iou = intersection / union if union > 0 else 1.0

        return {
            "dice": float(dice),
            "iou": float(iou),
            "voxels_A": int(volA),
            "voxels_B": int(volB),
            "intersection": int(intersection),
        }


    def computeVolumeMetrics(self, voxelsA: int, voxelsB: int, pitch: float) -> dict:
        voxelVolume = pitch ** 3
        volA = voxelsA * voxelVolume
        volB = voxelsB * voxelVolume

        relDiff = (volB - volA) / volA * 100 if volA > 0 else 0.0

        return {
            "volume_A_mm3": float(volA),
            "volume_B_mm3": float(volB),
            "relative_difference_percent": float(relDiff),
        }


    def rasterizeModelToVoxelGrid(
        self,
        modelNode,
        referenceOrigin,
        referenceSpacing,
        referenceExtent,
    ) -> np.ndarray:
        """
        Rasterize a model into a voxel grid defined by origin, spacing and extent.
        """
        modelNode.HardenTransform()
        polyData = modelNode.GetPolyData()

        image = vtk.vtkImageData()
        image.SetOrigin(referenceOrigin)
        image.SetSpacing(referenceSpacing)
        image.SetExtent(referenceExtent)
        image.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
        image.GetPointData().GetScalars().Fill(0)

        polyToStencil = vtk.vtkPolyDataToImageStencil()
        polyToStencil.SetInputData(polyData)
        polyToStencil.SetOutputOrigin(referenceOrigin)
        polyToStencil.SetOutputSpacing(referenceSpacing)
        polyToStencil.SetOutputWholeExtent(referenceExtent)
        polyToStencil.Update()

        stencil = vtk.vtkImageStencil()
        stencil.SetInputData(image)
        stencil.SetStencilConnection(polyToStencil.GetOutputPort())
        stencil.ReverseStencilOff()
        stencil.SetBackgroundValue(0)
        stencil.Update()

        img = stencil.GetOutput()
        arr = vtk_np.vtk_to_numpy(img.GetPointData().GetScalars())
        arr = arr.reshape(
            img.GetDimensions()[::-1]
        )  # z, y, x

        return arr.astype(bool)


    def voxelizeModelAndCompare(
        self,
        inputModel: vtkMRMLModelNode,
        originalSegmentationModel: vtkMRMLModelNode,
        outputVoxelModel: vtkMRMLModelNode,
        pitch: float,
        ui=None,
    ) -> dict:
        """
        Voxelize mesh, rasterize original segmentation,
        compute Dice, IoU and volume metrics.
        """

        # ---------------------------
        # 1. Trimesh voxelization
        # ---------------------------
        inputModel.HardenTransform()
        polyData = inputModel.GetPolyData()

        points = vtk_np.vtk_to_numpy(polyData.GetPoints().GetData())
        cells = vtk_np.vtk_to_numpy(polyData.GetPolys().GetData())
        faces = cells.reshape(-1, 4)[:, 1:]

        mesh = trimesh.Trimesh(vertices=points, faces=faces)
        voxelized = mesh.voxelized(pitch=pitch).fill()
        voxelMask = voxelized.matrix.astype(bool)

        # ---------------------------
        # 2. Reference grid
        # ---------------------------
        origin = voxelized.origin
        spacing = (pitch, pitch, pitch)
        dims = voxelMask.shape
        extent = (0, dims[2] - 1, 0, dims[1] - 1, 0, dims[0] - 1)

        # ---------------------------
        # 3. Rasterize original segmentation
        # ---------------------------
        originalMask = self.rasterizeModelToVoxelGrid(
            originalSegmentationModel,
            referenceOrigin=origin,
            referenceSpacing=spacing,
            referenceExtent=extent,
        )

        # ---------------------------
        # 4. Metrics
        # ---------------------------
        voxelMetrics = self.computeVoxelMetrics(originalMask, voxelMask)
        volumeMetrics = self.computeVolumeMetrics(
            voxelMetrics["voxels_A"],
            voxelMetrics["voxels_B"],
            pitch,
        )

        # ---------------------------
        # 5. Output voxel model
        # ---------------------------
        if voxelized.matrix.sum() < 50_000:
            surface_mesh = voxelized.as_boxes()
        else:
            slicer.util.warningDisplay(
                "Voxel mesh too dense to visualize safely.\nMetrics computed anyway."
            )

        out_poly = vtk.vtkPolyData()
        v_vtk = vtk_np.numpy_to_vtk(surface_mesh.vertices, deep=True)
        pts = vtk.vtkPoints()
        pts.SetData(v_vtk)
        out_poly.SetPoints(pts)

        faces_out = surface_mesh.faces
        num_faces = faces_out.shape[0]
        cells_array = np.hstack(
            [np.full((num_faces, 1), 3), faces_out]
        ).astype(np.int64)

        cells_vtk = vtk_np.numpy_to_vtkIdTypeArray(cells_array, deep=True)
        connectivity = vtk.vtkCellArray()
        connectivity.SetCells(num_faces, cells_vtk)
        out_poly.SetPolys(connectivity)

        outputVoxelModel.SetAndObservePolyData(out_poly)
        outputVoxelModel.CreateDefaultDisplayNodes()
        outputVoxelModel.GetDisplayNode().SetVisibility(True)

        if ui:
            ui.voxelCountOriginal.setText(f"{volumeMetrics['volume_A_mm3']}")
            ui.voxelCountNew.setText(f"{volumeMetrics['volume_B_mm3']}")
            ui.diceLabel.setText(f"{voxelMetrics['dice']:.3f}")
            ui.iouLabel.setText(f"{voxelMetrics['iou']:.3f}")
            ui.volumeDiffLabel.setText(
                f"{volumeMetrics['relative_difference_percent']:+.2f}%"
            )
            slicer.app.processEvents()

        return {
            **voxelMetrics,
            **volumeMetrics,
        }

    def voxelizeModelToModel(self, inputModel: vtkMRMLModelNode, outputModel: vtkMRMLModelNode, 
                                pitch: float, ui=None,) -> None:
            
            ###Voxelizes the entire model using a given pitch.
            ###:param pitch: float
            

            if not inputModel or not outputModel:
                raise ValueError("Invalid input or output model")
            
            # Convert Slicer MRML model to Trimesh
            inputModel.HardenTransform()
            polyData = inputModel.GetPolyData()
            
            # Extract vertices and faces from VTK PolyData
            points = vtk.util.numpy_support.vtk_to_numpy(polyData.GetPoints().GetData())
            cells = vtk.util.numpy_support.vtk_to_numpy(polyData.GetPolys().GetData())
            
            # VTK polys are stored as [n, id1, id2, id3, n, id4...] triangles
            faces = cells.reshape(-1, 4)[:, 1:]

            mesh = trimesh.Trimesh(vertices=points, faces=faces)

            # Voxelize and fill: create the occupancy grid and fill the interior
            voxelized = mesh.voxelized(pitch=pitch).fill()

            # The 'matrix' is a boolean array where True represents a filled voxel
            totalVoxelCount = np.sum(voxelized.matrix)

            if ui:
                ui.voxelCount.setText(f"{totalVoxelCount:,}") 
            
                # Ensure the UI refreshes immediately
                slicer.app.processEvents()
            
            # Convert back to geometry: create a single mesh made of cubes
            surface_mesh = voxelized.as_boxes()

            # Convert Trimesh back to VTK PolyData for Slicer
            v_out = surface_mesh.vertices
            f_out = surface_mesh.faces
            
            # Create VTK object
            out_poly = vtk.vtkPolyData()
            
            # Set points
            v_vtk = vtk.util.numpy_support.numpy_to_vtk(v_out, deep=True)
            pts = vtk.vtkPoints()
            pts.SetData(v_vtk)
            out_poly.SetPoints(pts)
            
            # Set cells as triangles for the cube faces
            num_faces = f_out.shape[0]
            # VTK expects [3, id1, id2, id3, 3, id4...]
            cells_array = np.hstack([np.full((num_faces, 1), 3), f_out]).astype(np.int64)
            cells_vtk = vtk.util.numpy_support.numpy_to_vtkIdTypeArray(cells_array, deep=True)
            
            connectivity = vtk.vtkCellArray()
            connectivity.SetCells(num_faces, cells_vtk)
            out_poly.SetPolys(connectivity)

            # Update Slicer Model Node
            outputModel.SetAndObservePolyData(out_poly)
            
            if not outputModel.GetDisplayNode():
                outputModel.CreateDefaultDisplayNodes()
            
            outputModel.GetDisplayNode().SetVisibility(True)
            outputModel.GetPolyData().Modified()
                    
            
            
            
            
            
            
            
    def scaleModel(self, inputModel, outputModel, scaleFactor=1.0):
            """
            Scales the entire model by a given factor.
            :param scaleFactor: float (e.g., 2.0 to double size, 0.5 to shrink)
            """
            if not inputModel or not outputModel:
                    raise ValueError("Invalid input or output nodes.")

            logging.info(f"Scaling model by factor: {scaleFactor}")
                
            # Get the polydata from the input model
            inputPolyData = inputModel.GetPolyData()
                
            # Create a transform and set the scale
            transform = vtk.vtkTransform()
            transform.Scale(scaleFactor, scaleFactor, scaleFactor)
                
            # Apply the transform to the polydata
            transformFilter = vtk.vtkTransformPolyDataFilter()
            transformFilter.SetInputData(inputPolyData)
            transformFilter.SetTransform(transform)
            transformFilter.Update()
                
            # Update the output model
            outputModel.SetAndObservePolyData(transformFilter.GetOutput())
                
            # Ensure it shows up in views
            outputModel.CreateDefaultDisplayNodes()
            outputModel.Modified()
                
            logging.info("Scaling complete.")
        

    def exportModelVTK(self, modelNode, filePath):
        """Writes the polydata to a VTK (.vtk) file."""
        if not modelNode:
            return
        
        writer = vtk.vtkPolyDataWriter()
        writer.SetFileName(filePath)
        writer.SetInputData(modelNode.GetPolyData())
        writer.Write()

    def exportModelSTL(self, modelNode, filePath):
        """Writes the polydata to an STL (.stl) file."""
        if not modelNode:
            return
            
        writer = vtk.vtkSTLWriter()
        writer.SetFileName(filePath)
        writer.SetInputData(modelNode.GetPolyData())

        # STL files usually require binary format for smaller file size
        writer.SetFileTypeToBinary()
        writer.Write()

    def exportModelMSH(self, modelNode, filePath):
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
        logging.info(f"MSH (Version 2.2 ASCII) saved to {filePath}")
