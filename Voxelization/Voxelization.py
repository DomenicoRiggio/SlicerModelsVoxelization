import os
from typing import Optional

import vtk

import slicer
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import parameterNodeWrapper

from slicer import vtkMRMLModelNode, vtkMRMLScalarVolumeNode


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
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "Utilities")]
        self.parent.dependencies = []  # TODO: add here list of module names that this module requires
        self.parent.contributors = ["Laura Lichtlein (KIT Institute of Biomedical Engineering)", "Domenico Riggio (KIT Institute of Biomedical Engineering)", "Ciro Benito Raggio (KIT Institute of Biomedical Engineering)"]
        self.parent.helpText = _("""
This module provides tools for 3D model manipulation and export. Features include uniformly resizing models, 
converting surface meshes into solid cubical voxel models and exporting processed the models to a chosen 
directory in .vtk, .stl, and .msh (Gmsh 2.2 ASCII) formats.
""")
        self.parent.acknowledgementText = _("""
This module was developed by Laura Lichtlein, Domenico Riggio, Ciro Benito Raggio (KIT Institute of Biomedical Engingeering).
""")




#
# VoxelizationParameterNode
#


@parameterNodeWrapper
class VoxelizationParameterNode:
    """
    The parameters needed by module.

    inputVolume - The VTK reference scalar volume
    inputModel  - The VTK model to be voxelized.
    pitch       - The value setting the side length for the voxels.
    outputModel  - The VTK model to be exported.
    """
    inputVolume: vtkMRMLScalarVolumeNode
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
 
        if self._parameterNode and not self._parameterNode.inputVolume:
            firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
            if firstVolumeNode:
                self._parameterNode.inputVolume = firstVolumeNode
                
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
    
    def setInfoLabel(self, text):
        self.ui.infoLabel.setText(text)

    def _checkCanApply(self, caller=None, event=None) -> None:
        if self._parameterNode and self._parameterNode.inputVolume and self._parameterNode.inputModel and self._parameterNode.outputModel:
            self.ui.exportToFileButton.enabled = True
            self.ui.exportToFileButton.toolTip = _("Export model")

            self.ui.voxelButton.enabled = True
            self.ui.voxelButton.toolTip = _("Voxelize model")
        else:
            self.ui.exportToFileButton.enabled = False
            self.ui.exportToFileButton.toolTip = _("Select input and output models")

            self.ui.voxelButton.enabled = False
            self.ui.voxelButton.toolTip = _("Select input and output models")

    def onVoxelButton(self) -> None:
        """Voxelize the input model"""

        with slicer.util.tryWithErrorDisplay(_("Voxelization failed."), waitCursor=True):
            self.setInfoLabel("")
            inputVolume = self.ui.inputVolumeSelector.currentNode()
            inputModel = self.ui.inputModelSelector.currentNode()
            outputModel = self.ui.outputSelectorModel.currentNode()
            
            pitch = float(self.ui.pitchWidget.value)
            

            with slicer.util.tryWithErrorDisplay(_("Operation failed."), waitCursor=True):
                self.logic.voxelizeModelToModel(inputVolume, inputModel, outputModel, pitch, self.ui)
            
            self.setInfoLabel("Processing completed.")
                
    def onExportButton(self) -> None:
        """Export output model to file"""
        self.setInfoLabel("")
        outputModel = self.ui.outputSelectorModel.currentNode()
        directory = self.ui.DirectoryButton.directory

        if not outputModel:
            slicer.util.errorDisplay("Please select an Output Model first.")
            return
        if not directory:
            slicer.util.errorDisplay("Please select a save directory.")
            return

        baseFileName = outputModel.GetName()
        paths = {
            "VTK": os.path.join(directory, f"{baseFileName}.vtk"),
            "STL": os.path.join(directory, f"{baseFileName}.stl"),
            "MSH": os.path.join(directory, f"{baseFileName}.msh"),
        }

        selectedFormat = self.ui.exportFormatCombo.currentText

        if selectedFormat not in paths:
            slicer.util.errorDisplay(f"Unknown export format: {selectedFormat}")
            return

        with slicer.util.tryWithErrorDisplay(_("Export failed."), waitCursor=True):
            export_map = {
                "VTK": self.logic.exportModelVTK,
                "STL": self.logic.exportModelSTL,
                "MSH": self.logic.exportModelMSH,
            }
            export_func = export_map[selectedFormat]
            export_func(outputModel, paths[selectedFormat])

            self.setInfoLabel(f"Model saved to:{paths[selectedFormat]}")


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

    
    def voxelizeModelToModel(self, 
                            inputVolume: vtkMRMLScalarVolumeNode,
                            inputModel: vtkMRMLModelNode, 
                            outputModel: vtkMRMLModelNode, 
                            pitch: float, 
                            ui=None,) -> None:

        from VoxelizationLib.logicUtils import rasterizeModelToVolume, getVoxelizedModel, displayVoxelizedModel, computeMetrics
        from numpy import count_nonzero
        
        if not inputVolume:
            raise ValueError("Invalid input volume")
        
        if not inputModel:
            raise ValueError("Invalid input model")
        
        if not outputModel:
            raise ValueError("Invalid output model")
        
        mask_orig = rasterizeModelToVolume(inputModel, inputVolume)
        originalVoxelCount = int(count_nonzero(mask_orig))
        grid_original = mask_orig != 0

        voxelizedModel = getVoxelizedModel(inputModel, pitch, outputModel)

        displayVoxelizedModel(voxelizedModel)


        mask = rasterizeModelToVolume(voxelizedModel, inputVolume)
        voxelizedVoxelCount = int(count_nonzero(mask))
        grid_voxelized = mask != 0

        metrics = computeMetrics(grid_original, grid_voxelized, originalVoxelCount, voxelizedVoxelCount)
        
        dice = metrics["dice"]
        iou = metrics["iou"]
        deltaV = metrics["deltaV"]
        
        if ui:
            ui.voxelCountOriginal.setText(f"{originalVoxelCount}")
            ui.voxelCountNew.setText(f"{voxelizedVoxelCount}")
            ui.diceScore.setText(f"{dice:.6f}")
            ui.iouScore.setText(f"{iou:.6f}")
            ui.deltaV.setText(f"{deltaV:.6f}")
            slicer.app.processEvents()
        

    def exportModelVTK(self, modelNode, filePath):
        from VoxelizationLib.logicUtils import exportModelVTK
        exportModelVTK(modelNode, filePath)

    def exportModelSTL(modelNode, filePath):
        from VoxelizationLib.logicUtils import exportModelSTL
        exportModelSTL(modelNode, filePath)
        
    def exportModelMSH(modelNode, filePath):
        from VoxelizationLib.logicUtils import exportModelMSH
        exportModelMSH(modelNode, filePath)
