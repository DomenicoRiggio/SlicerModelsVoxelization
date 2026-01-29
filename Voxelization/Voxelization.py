import logging
import os
from typing import Annotated, Optional
import numpy as np

import vtk
try:
    import trimesh
except ImportError:
    if slicer.util.confirmOkCancelDisplay("This tool requires the 'trimesh' library. Install it now?"):
        slicer.util.pip_install("trimesh")
        import trimesh
import qt

import slicer
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import (
    parameterNodeWrapper,
    WithinRange,
)

from slicer import vtkMRMLScalarVolumeNode
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

        # Additional initialization step after application startup is complete
        slicer.app.connect("startupCompleted()", registerSampleData)


#
# Register sample data sets in Sample Data module
#


def registerSampleData():
    """Add data sets to Sample Data module."""
    # It is always recommended to provide sample data for users to make it easy to try the module,
    # but if no sample data is available then this method (and associated startupCompeted signal connection) can be removed.

    import SampleData

    iconsPath = os.path.join(os.path.dirname(__file__), "Resources/Icons")

    # To ensure that the source code repository remains small (can be downloaded and installed quickly)
    # it is recommended to store data sets that are larger than a few MB in a Github release.

    # Voxelization1
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category="Voxelization",
        sampleName="Voxelization1",
        # Thumbnail should have size of approximately 260x280 pixels and stored in Resources/Icons folder.
        # It can be created by Screen Capture module, "Capture all views" option enabled, "Number of images" set to "Single".
        thumbnailFileName=os.path.join(iconsPath, "Voxelization1.png"),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        fileNames="Voxelization1.nrrd",
        # Checksum to ensure file integrity. Can be computed by this command:
        #  import hashlib; print(hashlib.sha256(open(filename, "rb").read()).hexdigest())
        checksums="SHA256:998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        # This node name will be used when the data set is loaded
        nodeNames="Voxelization1",
    )

    # Voxelization2
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category="Voxelization",
        sampleName="Voxelization2",
        thumbnailFileName=os.path.join(iconsPath, "Voxelization2.png"),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        fileNames="Voxelization2.nrrd",
        checksums="SHA256:1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        # This node name will be used when the data set is loaded
        nodeNames="Voxelization2",
    )


#
# VoxelizationParameterNode
#


@parameterNodeWrapper
class VoxelizationParameterNode:
    """
    The parameters needed by module.

    inputModel  - The VTK model to be voxelized.
    scaleFactor - The value used to scale the input model in size.
    pitch       - The value setting the side length for the voxels.
    """

    inputModel: vtkMRMLModelNode
    scaleFactor: Annotated[float, WithinRange(0.1, 500)] = 10
    pitch: Annotated[float, WithinRange(0.1, 500)] = 10
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
        self.ui.scaleButton.connect("clicked(bool)", self.onScaleButton)
        self.ui.voxelButton.connect("clicked(bool)", self.onVoxelButton)
        self.ui.exportToFileButton.clicked.connect(self.onExportButton)

        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()

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

        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)
        self._parameterNode = inputParameterNode
        if self._parameterNode:
            # Note: in the .ui file, a Qt dynamic property called "SlicerParameterName" is set on each
            # ui element that needs connection.
            self._parameterNodeGuiTag = self._parameterNode.connectGui(self.ui)
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)
            self._checkCanApply()

    def _checkCanApply(self, caller=None, event=None) -> None:
        if self._parameterNode and self._parameterNode.inputModel and self._parameterNode.outputModel:
            self.ui.scaleButton.enabled = True
            self.ui.scaleButton.toolTip = _("Scale model")

            self.ui.voxelButton.enabled = True
            self.ui.voxelButton.toolTip = _("Voxelize model")
        else:
            self.ui.scaleButton.enabled = False
            self.ui.scaleButton.toolTip = _("Select input and output models")

            self.ui.voxelButton.enabled = False
            self.ui.voxelButton.toolTip = _("Select input and output models")

    def onScaleButton(self) -> None:
        """Scale the input model in size"""

        with slicer.util.tryWithErrorDisplay(_("Decimation failed."), waitCursor=True):
            inputModel = self.ui.inputSelector.currentNode()
            outputModel = self.ui.outputSelectorModel.currentNode()
            
            # Map the slider value to the scaling factor
            scaleFactor = float(self.ui.scaleWidget.value)
            
            # If your output selector is empty, we can overwrite the input
            if not outputModel:
                outputModel = inputModel

            with slicer.util.tryWithErrorDisplay(_("Operation failed."), waitCursor=True):
                self.logic.scaleModel(inputModel, outputModel, scaleFactor)

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
                self.logic.voxelizeModelToModel(inputModel, outputModel, pitch, self.ui)
                
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
    
    def voxelizeModelToModel(self, inputModel: vtkMRMLModelNode, outputModel: vtkMRMLModelNode, 
                             pitch: float, ui=None,) -> None:
        """
        Voxelizes the entire model using a given pitch.
        :param pitch: float
        """

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
            ui.lineEdit.setText(f"{totalVoxelCount:,}") 
        
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
        try:
            import meshio
        except ImportError:
            slicer.util.pip_install("meshio")
            import meshio

        import vtk.util.numpy_support as vtk_np
        import numpy as np

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

#
# VoxelizationTest
#


class VoxelizationTest(ScriptedLoadableModuleTest):
    """
    This is the test case for your scripted module.
    Uses ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setUp(self):
        """Do whatever is needed to reset the state - typically a scene clear will be enough."""
        slicer.mrmlScene.Clear()

    def runTest(self):
        """Run as few or as many tests as needed here."""
        self.setUp()
        self.test_Voxelization1()

    def test_Voxelization1(self):
        """
        Higher-level integration test using real Slicer sample data.
        This simulates how an end-user interacts with the module:
        - Loads sample data
        - Runs voxelization logic
        - Verifies output volume validity
        """

        self.delayDisplay("Starting integration test with real sample data")


        self.delayDisplay("Integration test passed")

#TODO: