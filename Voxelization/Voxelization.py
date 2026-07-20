# Standard library imports
import os
import sys

# Ensure the module directory is in sys.path so VoxelizationLib can be imported
_moduleDir = os.path.dirname(os.path.abspath(__file__))
if _moduleDir not in sys.path:
    sys.path.insert(0, _moduleDir)
from typing import Optional

# VTK is the 3D rendering/geometry backbone used by 3D Slicer
import vtk

# Slicer core imports
import slicer
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import parameterNodeWrapper, WithinRange, Default
from typing import Annotated

from slicer import vtkMRMLModelNode, vtkMRMLScalarVolumeNode
from slicer import vtkMRMLSegmentationNode


# =============================================================================
# Voxelization  -  Module descriptor
# Registers the module in Slicer's module menu with metadata.
# =============================================================================
#
# Voxelization
#


# ScriptedLoadableModule provides the standard module registration API.
class Voxelization(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("Voxelization")
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "Utilities")]
        self.parent.dependencies = []
        self.parent.contributors = [
            "Laura Lichtlein (KIT Institute of Biomedical Engineering)",
            "Domenico Riggio (KIT Institute of Biomedical Engineering)",
            "Ciro Benito Raggio (KIT Institute of Biomedical Engineering)",
        ]
        self.parent.helpText = _("""
This module provides tools for 3D model manipulation and export. Features include uniformly resizing models,
converting surface meshes into solid cubical voxel models and exporting processed the models to a chosen
directory in .vtk, .stl, and .msh (Gmsh 2.2 ASCII) formats.
""")
        self.parent.acknowledgementText = _("""
This module was developed by Laura Lichtlein, Domenico Riggio, Ciro Benito Raggio
(KIT Institute of Biomedical Engineering).
""")

    def registerIO(self):
        """
        Called automatically by qSlicerScriptedLoadableModule during module
        registration. Registers the MSH file writer so ".msh" appears
        directly in Slicer's standard "Save Data" dialog.
        """
        from VoxelizationLib.MSHFileWriter import MSHFileWriter
        slicer.app.ioManager().registerIO(MSHFileWriter(self))


#
# VoxelizationParameterNode
#


# =============================================================================
# VoxelizationParameterNode  -  Typed, auto-serialised parameter storage
# @parameterNodeWrapper generates getters/setters for each field and connects
# them to the MRML scene so values survive scene save/load and undo/redo.
# =============================================================================
@parameterNodeWrapper
class VoxelizationParameterNode:
    """
    inputVolume       - The VTK reference scalar volume
    inputModel        - The model to be voxelized (Model mode)
    inputSegmentation - The segmentation node (Segmentation mode)
    pitch             - Side length for the voxels
    outputModel       - The output model node
    """
    inputVolume: vtkMRMLScalarVolumeNode
    inputModel: vtkMRMLModelNode
    inputSegmentation: vtkMRMLSegmentationNode
    pitch: Annotated[float, WithinRange(0.1, 1000.0), Default(0.5)]
    threshold: Annotated[float, WithinRange(0.0, 1.0), Default(0.5)]
    outputModel: vtkMRMLModelNode


#
# VoxelizationWidget
#


# =============================================================================
# VoxelizationWidget  -  GUI controller
# Inherits ScriptedLoadableModuleWidget (Slicer panel lifecycle) and
# VTKObservationMixin (safe VTK observer management with auto-cleanup).
# =============================================================================
class VoxelizationWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):

    def __init__(self, parent=None) -> None:
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)
        self.logic = None  # Business logic instance
        self._parameterNode = None  # Currently active parameter node
        self._parameterNodeGuiTag = None  # Tag returned by connectGui(); needed to disconnect later
        self.requiredDeps = ["trimesh", "meshio"]  # Python packages required at runtime
        self._segmentationObserverTag = None  # VTK observer tag on the currently watched segmentation node
        self._observedSegmentationNode = None  # Reference to the segmentation node being observed
        self._lastOutputNode    = None  # Last voxelized output node — used for export
        self._pendingOutputNode = None  # Node created via + button but not yet voxelized
        self._pendingOutputName = None  # Name captured from + button before node was deleted
        self._lastVoxName       = None  # voxName of the last voxelization — used to detect re-runs
        self._metricsStore   = {}    # {modelName: {metricKey: value}} — persists metrics per model

    # ------------------------------------------------------------------
    # Dependency check
    # ------------------------------------------------------------------

    def checkDependencies(self):
        from importlib.util import find_spec
        allPresent = all(find_spec(mod) is not None for mod in self.requiredDeps)  # find_spec returns None when the package is not installed

        if not allPresent:
            if not slicer.util.confirmOkCancelDisplay(
                "The dependencies needed for the extension will be installed. "
                "The operation may take a few minutes. A Slicer restart will be necessary.",
                "Press OK to install and restart.",
            ):
                raise ValueError("Missing dependencies.")

            slicer.util.setPythonConsoleVisible(True)
            print("Installing missing dependencies, please wait...")

            try:
                for dep in self.requiredDeps:
                    print(f"Installing {dep}...")
                    slicer.util.pip_install(dep)
                print("All dependencies installed successfully.")
                slicer.app.restart()
            except Exception as e:
                slicer.util.errorDisplay(f"Failed to install requirements: {e}")

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self) -> None:
        ScriptedLoadableModuleWidget.setup(self)

        uiWidget = slicer.util.loadUI(self.resourcePath("UI/Voxelization.ui"))  # Load the Qt Designer .ui file and embed it in the module panel
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)  # childWidgetVariables creates self.ui.<widgetName> for every named widget
        uiWidget.setMRMLScene(slicer.mrmlScene)  # Required so qMRMLNodeComboBox widgets know which scene to list nodes from

        self.logic = VoxelizationLogic()

        # Scene observers
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)  # Observe scene open/close to reset the parameter node accordingly
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        # Radio buttons — only need to connect one; toggled fires for both
        self.ui.modelRadioButton.toggled.connect(self.onInputTypeToggled)  # toggled fires for BOTH the checked and unchecked button - one connection is enough

        # When user picks a different segmentation node → refresh segment list
        self.ui.inputSegmentationSelector.currentNodeChanged.connect(self.onSegmentationNodeChanged)  # Rebuild the segment list whenever the user picks a different segmentation node
        self.ui.inputVolumeSelector.currentNodeChanged.connect(self._updateResolutionLabel)

        # Update the output name preview whenever the input selection changes
        self.ui.inputSegmentSelector.currentIndexChanged.connect(self._updateOutputNamePreview)
        self.ui.inputModelSelector.currentNodeChanged.connect(self._updateOutputNamePreview)

        # Action buttons
        self.ui.guideButton.connect("clicked(bool)", self.onGuideButton)
        self.ui.voxelButton.connect("clicked(bool)", self.onVoxelButton)

        # Set project logo next to the Guide button
        import qt, os
        logoPath = os.path.join(os.path.dirname(__file__), "Resources", "Icons", "Voxelization.png")
        if os.path.exists(logoPath):
            pix = qt.QPixmap(logoPath).scaledToHeight(40, qt.Qt.SmoothTransformation)
            self.ui.logoLabel.setPixmap(pix)
            self.ui.logoLabel.setFixedSize(pix.width(), pix.height())

        # Observe scene node additions and removals to keep all combos up to date
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.NodeAddedEvent,   self._onSceneNodeAdded)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.NodeRemovedEvent, self._onSceneNodeRemoved)
        self.ui.exportToFileButton.connect("clicked(bool)", self.onExportButton)
        self.ui.outputSelectorModel.currentNodeChanged.connect(self._onOutputSelectorChanged)
        self.ui.exportMetricsButton.connect("clicked(bool)", self.onExportMetricsButton)
        self.ui.metricsModelCombo.currentIndexChanged.connect(self._onMetricsModelChanged)
        self.ui.pitchMaxSpinBox.valueChanged.connect(self._onPitchMaxChanged)

        # Initialise the metrics table with row labels
        self._initMetricsTable()
        self.ui.booleanApplyButton.connect("clicked(bool)", self.onBooleanApplyButton)
        self.ui.booleanModelACombo.currentIndexChanged.connect(self._clearBooleanResult)
        self.ui.booleanModelBCombo.currentIndexChanged.connect(self._clearBooleanResult)
        self.ui.booleanOperationCombo.currentIndexChanged.connect(self._clearBooleanResult)

        # Populate boolean model combos when the boolean section is expanded
        self.ui.booleanCollapsibleButton.connect("contentsCollapsed(bool)", self.onBooleanSectionToggled)
        self.ui.outputsCollapsibleButton.connect("contentsCollapsed(bool)", self.onExportSectionToggled)

        # Boolean section starts disabled — enabled once 2+ voxelized models exist
        self._updateBooleanSectionState()

        # Start in Model mode
        self.ui.segmentationRadioButton.setChecked(True)  # Default to Segmentation mode on first open
        self._updateInputVisibility()

        self.initializeParameterNode()

        # Enforce pitch max on slider after full UI init
        try:
            self.ui.pitchWidget.maximum = float(self.ui.pitchMaxSpinBox.value)
        except Exception:
            pass

        self.checkDependencies()

    # ------------------------------------------------------------------
    # Input-type toggle
    # ------------------------------------------------------------------

    def _onOutputSelectorChanged(self, node):
        """
        Called when the output selector changes.
        If a new empty node was created via the + button, track it as
        pending. Do NOT delete it — the user may have named it intentionally.
        """
        if node and node.GetPolyData() and node.GetPolyData().GetNumberOfPoints() == 0:
            self._pendingOutputNode = node
            self._pendingOutputName = node.GetName()
        else:
            # User selected an existing node with data — clear pending
            if node and node != self._pendingOutputNode:
                self._pendingOutputNode = None
                self._pendingOutputName = None

    def _cleanupPendingOutputNode(self):
        """Remove the pending empty output node if it was never voxelized."""
        if self._pendingOutputNode:
            try:
                node = self._pendingOutputNode
                # Only remove if still empty (never voxelized)
                if node and node.GetPolyData() and node.GetPolyData().GetNumberOfPoints() == 0:
                    slicer.mrmlScene.RemoveNode(node)
            except Exception:
                pass
            self._pendingOutputNode = None
            self._pendingOutputName = None
        """Called whenever the Model/Segmentation radio changes."""
        self._updateInputVisibility()
        self._checkCanApply()

    def _updateResolutionLabel(self, node=None):
        """
        Show the voxel spacing of the selected input volume as X × Y × Z mm.
        Called whenever the user picks a different volume node.
        """
        if node is None:
            node = self.ui.inputVolumeSelector.currentNode()

        if not node:
            self.ui.resolutionValueLabel.setText("—")
            return

        spacing = node.GetSpacing()  # returns (x, y, z) in mm
        self.ui.resolutionValueLabel.setText(
            f"{spacing[0]:.3f} × {spacing[1]:.3f} × {spacing[2]:.3f}"
        )
        """Called whenever the Model/Segmentation radio changes."""
        self._updateInputVisibility()
        self._checkCanApply()

    def onInputTypeToggled(self, modelChecked):
        """Called whenever the Model/Segmentation radio changes."""
        self._updateInputVisibility()
        self._checkCanApply()

    def _onPitchMaxChanged(self, value):
        """
        Update the pitch slider maximum when the user changes the spinbox.
        """
        self.ui.pitchWidget.maximum    = value
        self.ui.pitchWidget.singleStep = max(0.05, round(value * 0.05, 2))
        # Clamp current value if it exceeds the new max
        currentVal = self.ui.pitchWidget.value
        if currentVal > value:
            self.ui.pitchWidget.value = value

    def _updateInputVisibility(self):
        """Show Model row OR Segmentation + Segment rows — never both."""
        isModel = self.ui.modelRadioButton.isChecked()
  # Show Model row (label + node selector showing only vtkMRMLModelNode)
        # Model row
        self.ui.inputModelLabel.setVisible(isModel)
        self.ui.inputModelSelector.setVisible(isModel)

        # Segmentation rows
        # NOTE: widget names must match exactly what is in Voxelization.ui  # NOTE: widget names must match exactly what is declared in Voxelization.ui
        self.ui.inputSegmentionLabel.setVisible(not isModel)       # "Input segmentation:"  # 'Input segmentation:' label
        self.ui.inputSegmentationSelector.setVisible(not isModel)  # qMRMLNodeComboBox  # qMRMLNodeComboBox filtered to vtkMRMLSegmentationNode
        self.ui.inputSegmentLabel.setVisible(not isModel)          # "Segment:"  # 'Segment:' label
        self.ui.inputSegmentSelector.setVisible(not isModel)       # QComboBox (segment names)  # Plain QComboBox populated dynamically from Python

    # ------------------------------------------------------------------
    # Segment combo population
    # ------------------------------------------------------------------

    def onSegmentationNodeChanged(self, node):
        """
        Called when the user selects a different segmentation node.
        Repopulates inputSegmentSelector with that node's segment names.
        """
        # Stop watching the old node
        if self._observedSegmentationNode and self._segmentationObserverTag:  # Detach VTK observer from the old node to prevent memory leaks
            self._observedSegmentationNode.GetSegmentation().RemoveObserver(
                self._segmentationObserverTag
            )
            self._segmentationObserverTag = None
            self._observedSegmentationNode = None

        self._populateSegmentCombo(node)

        # If no node selected, clear the combo explicitly
        if not node:
            self.ui.inputSegmentSelector.clear()
            self.ui.outputSelectorModel.setCurrentNode(None)
            self._lastOutputNode = None

        # Watch the new node so the list stays up to date
        if node:
            self._observedSegmentationNode = node
            self._segmentationObserverTag = node.GetSegmentation().AddObserver(
                vtk.vtkCommand.ModifiedEvent, self._onSegmentationModified
            )

        self._checkCanApply()

    def _onSegmentationModified(self, caller, event):
        """Refresh the segment list when segments are added or removed."""
        self._populateSegmentCombo(self._observedSegmentationNode)
        self._checkCanApply()

    def _updateOutputNamePreview(self, *args):
        """
        Show a preview of the output name in the info label.
        Does NOT create any node — the node is created only when Apply is clicked.
        """
        isModel = self.ui.modelRadioButton.isChecked()

        if isModel:
            inputNode = self.ui.inputModelSelector.currentNode()
            if not inputNode:
                self.ui.outputSelectorModel.setCurrentNode(None)
                self._lastOutputNode = None
                self.setInfoLabel("")
                return
            baseName = f"{inputNode.GetName()}_vox"
        else:
            segNode = self.ui.inputSegmentationSelector.currentNode()
            segName = self.ui.inputSegmentSelector.currentText
            if not segNode or not segName:
                self.ui.outputSelectorModel.setCurrentNode(None)
                self._lastOutputNode = None
                self.setInfoLabel("")
                return
            baseName = f"{segName}_vox"

        # Find the unique name without creating anything
        uniqueName = baseName
        counter    = 1
        while slicer.mrmlScene.GetFirstNodeByName(uniqueName):
            uniqueName = f"{baseName}_{counter}"
            counter   += 1

        # Clean up any pending empty node from a previous + click
        self._cleanupPendingOutputNode()
        # Reset last vox name so a new node is created for this input
        self._lastVoxName       = None
        self._pendingOutputName = None

        # Clear any stale output node
        currentNode = self.ui.outputSelectorModel.currentNode()
        if currentNode and currentNode.GetPolyData() and currentNode.GetPolyData().GetNumberOfPoints() == 0:
            slicer.mrmlScene.RemoveNode(currentNode)

        self.ui.outputSelectorModel.setCurrentNode(None)
        self._lastOutputNode = None
        self.setInfoLabel(f"Output will be: {uniqueName}")

    def _populateSegmentCombo(self, segmentationNode):
        """
        Fill inputSegmentSelector (QComboBox) with segments from segmentationNode.
        Item text  = human-readable segment name
        Item data  = segment ID (used internally; stable even if names are duplicated)
        """
        combo = self.ui.inputSegmentSelector
        combo.clear()  # combo.clear() empties the list before repopulating

        if segmentationNode is None:
            return

        segmentation = segmentationNode.GetSegmentation()
        for i in range(segmentation.GetNumberOfSegments()):  # addItem(display_text, user_data): name shown to user, ID stored internally
            segId   = segmentation.GetNthSegmentID(i)
            segName = segmentation.GetSegment(segId).GetName()
            combo.addItem(segName, segId)   # addItem(display_text, user_data)  # Using ID as data avoids bugs when two segments share a name

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def cleanup(self) -> None:
        self._cleanupPendingOutputNode()
        if self._observedSegmentationNode and self._segmentationObserverTag:
            self._observedSegmentationNode.GetSegmentation().RemoveObserver(
                self._segmentationObserverTag
            )
        self.removeObservers()

    def enter(self) -> None:
        self.initializeParameterNode()

    def exit(self) -> None:
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self._parameterNodeGuiTag = None
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)

    def onSceneStartClose(self, caller, event) -> None:
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event) -> None:
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self) -> None:
        self.setParameterNode(self.logic.getParameterNode())

        if self._parameterNode and not self._parameterNode.inputVolume:  # Pre-select the first available volume to save the user a click
            firstVolume = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
            if firstVolume:
                self._parameterNode.inputVolume = firstVolume

        # Update resolution display for the auto-selected volume
        self._updateResolutionLabel()

        # Clear any leftover output so the selector always starts empty.  # Output node is created on Apply - intentionally left empty here
        if self._parameterNode:
            self._parameterNode.outputModel = None

    def setParameterNode(self, inputParameterNode: Optional[VoxelizationParameterNode]) -> None:
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)

        self._parameterNode = inputParameterNode
        if self._parameterNode:
            self._parameterNodeGuiTag = self._parameterNode.connectGui(self.ui)
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)
            self._checkCanApply()
            # connectGui sets slider max from WithinRange — override with spinbox value
            try:
                self.ui.pitchWidget.maximum = float(self.ui.pitchMaxSpinBox.value)
            except Exception:
                pass

    def setInfoLabel(self, text):
        self.ui.infoLabel.setText(text)

    # ------------------------------------------------------------------
    # Apply-button guard
    # ------------------------------------------------------------------

    def _checkCanApply(self, caller=None, event=None) -> None:
        if self._parameterNode is None:
            self.ui.voxelButton.enabled = False
            self.ui.exportToFileButton.enabled = False
            return

        isModel = self.ui.modelRadioButton.isChecked()

        if isModel:
            hasInput = self.ui.inputModelSelector.currentNode() is not None  # Both a segmentation node AND at least one segment must be present
        else:
            # Need both a segmentation node AND at least one segment chosen
            hasInput = (
                self.ui.inputSegmentationSelector.currentNode() is not None
                and self.ui.inputSegmentSelector.count > 0
            )

        hasVolume = self._parameterNode.inputVolume is not None
        # Output node is created automatically on Apply — not required upfront  # Output node is created automatically on Apply - not required upfront
        canRun    = hasInput and hasVolume

        self.ui.voxelButton.enabled = canRun
        self.ui.voxelButton.toolTip = (
            _("Voxelize") if canRun
            else _("Select input (model or segmentation + segment), volume, and output model")
        )
        self.ui.exportToFileButton.enabled = canRun
        self.ui.exportToFileButton.toolTip = (
            _("Export model") if canRun else _("Select input and output first")
        )

    # ------------------------------------------------------------------
    # Voxelize button
    # ------------------------------------------------------------------

    def _onSceneNodeAdded(self, caller, event):
        """Refresh all model-listing combos when a new node is added to the scene."""
        self._refreshAllCombos()

    def _onSceneNodeRemoved(self, caller, event):
        """Refresh all model-listing combos when a node is removed from the scene."""
        self._refreshAllCombos()

    def _refreshAllCombos(self):
        """
        Refresh all plain QComboBox widgets that list models from the scene.
        Called whenever nodes are added or removed so deleted models
        disappear immediately from all dropdowns.
        """
        self._populateBooleanCombos()
        self._populateExportCombo()
        self._populateMetricsModelCombo()
        self._updateBooleanSectionState()

    def onGuideButton(self) -> None:
        """Show the Help Guide and Acknowledgements dialog."""
        from VoxelizationLib.UI.HelpDialog import HelpDialog
        dialog = HelpDialog(slicer.util.mainWindow())
        dialog.exec_()

    def onVoxelButton(self) -> None:
        with slicer.util.tryWithErrorDisplay(_("Voxelization failed."), waitCursor=True):
            self.setInfoLabel("")

            # Clear previous output so the selector doesn't carry a stale node
            self._parameterNode.outputModel = None
            # Reset pending tracker since Apply is now running
            self._pendingOutputNode = None

            inputVolume = self.ui.inputVolumeSelector.currentNode()
            pitch       = float(self.ui.pitchWidget.value)
            threshold   = float(self.ui.thresholdWidget.value)
            isModel     = self.ui.modelRadioButton.isChecked()

            if not inputVolume:
                raise ValueError("Please select an input volume.")

            # Warn for very small pitch values — computation grows as pitch^-3
            if pitch < 1.0:
                msg = (
                    f"Edge size {pitch:.2f} mm is very small and may cause a long computation "
                    f"or freeze the application. Do you want to proceed?"
                )
                if not slicer.util.confirmOkCancelDisplay(msg, "Warning: Small pitch value"):
                    return

            tempMesh  = None

            if isModel:
                # ---- Model mode: trimesh voxelization ----
                inputMesh      = self.ui.inputModelSelector.currentNode()
                if not inputMesh:
                    raise ValueError("Please select an input model.")
                voxName        = f"{inputMesh.GetName()}_vox"
                isSegmentation = False

            else:
                # ---- Segmentation mode ----
                segNode = self.ui.inputSegmentationSelector.currentNode()
                if not segNode:
                    raise ValueError("Please select a segmentation node.")

                idx = self.ui.inputSegmentSelector.currentIndex
                if idx < 0:
                    raise ValueError("Please select a segment from the list.")

                segmentId   = self.ui.inputSegmentSelector.itemData(idx)
                segmentName = self.ui.inputSegmentSelector.currentText

                inputMesh      = self.logic.segmentationToModel(segNode, segmentId)
                tempMesh       = inputMesh
                voxName        = f"{segmentName}_vox"
                isSegmentation = False

            # Determine the output node to use:
            existingNode = self.ui.outputSelectorModel.currentNode()

            def hasGeometry(node):
                if not node:
                    return False
                pd = node.GetPolyData()
                return pd is not None and pd.GetNumberOfPoints() > 0

            def makeUniqueName(baseName, excludeNode=None):
                name    = baseName
                counter = 1
                while True:
                    conflict = slicer.mrmlScene.GetFirstNodeByName(name)
                    if not conflict or (excludeNode and conflict.GetID() == excludeNode.GetID()):
                        break
                    name = f"{baseName}_{counter}"
                    counter += 1
                return name

            if existingNode:
                # User has selected a node — always use it regardless of content
                existingName = existingNode.GetName()
                if not existingName.endswith("_vox"):
                    uniqueName = makeUniqueName(f"{existingName}_vox", existingNode)
                    existingNode.SetName(uniqueName)
                else:
                    uniqueName = existingName
                outputModel = existingNode
                self._pendingOutputNode = None

            elif self._lastOutputNode and self._lastVoxName == voxName:
                # Same input as last run — overwrite
                outputModel = self._lastOutputNode
                uniqueName  = outputModel.GetName()

            else:
                # No node selected — create fresh
                uniqueName  = makeUniqueName(voxName)
                outputModel = self.logic._initModelNode(uniqueName)

            self._lastVoxName = voxName

            # If user typed a custom name via +, use that instead
            if self._pendingOutputName:
                customName = self._pendingOutputName
                if not customName.endswith("_vox"):
                    customName = f"{customName}_vox"
                # Check uniqueness
                counter = 1
                baseName = customName
                while slicer.mrmlScene.GetFirstNodeByName(customName):
                    customName = f"{baseName}_{counter}"
                    counter   += 1
                outputModel.SetName(customName)
                uniqueName = customName
                self._pendingOutputName = None

            self._parameterNode.outputModel = outputModel

            metricsValues = self.logic.voxelizeModelToModel(
                inputVolume, inputMesh, outputModel, pitch, threshold, self.ui
            )

            # Store metrics per model name so switching the combo reloads them
            if metricsValues:
                self._metricsStore[uniqueName] = metricsValues

            # Remove the temporary segment model from the scene
            if tempMesh is not None:
                slicer.mrmlScene.RemoveNode(tempMesh)

            # Place the result inside a "VoxelizedModels" folder in Subject Hierarchy
            self.logic.moveNodeToFolder(outputModel, "VoxelizedModels")

            # Store for export and show in selector
            self._lastOutputNode = outputModel
            self.ui.outputSelectorModel.setCurrentNode(outputModel)

            # Refresh boolean combos so new model appears there immediately
            self._populateBooleanCombos()
            self._updateBooleanSectionState()

            # Refresh export and metrics combos
            self._populateExportCombo(selectName=uniqueName)
            self._populateMetricsModelCombo(selectName=uniqueName)

            # Update the table LAST so nothing can overwrite it
            if metricsValues:
                self._updateMetricsTable(metricsValues)

            self.setInfoLabel(f"Processing completed. Output: {uniqueName}")

    # ------------------------------------------------------------------
    # Boolean operations
    # ------------------------------------------------------------------

    def onBooleanSectionToggled(self, collapsed):
        """Refresh the model combos every time the section is expanded."""
        if not collapsed:
            self._populateBooleanCombos()

    def _updateBooleanSectionState(self):
        """
        Enable the Boolean Operations section only when at least 2 voxelized
        models exist in the VoxelizedModels folder.
        Disabled with a tooltip explaining why when there are fewer.
        """
        shNode   = slicer.mrmlScene.GetSubjectHierarchyNode()
        sceneId  = shNode.GetSceneItemID()
        folderId = shNode.GetItemChildWithName(sceneId, "VoxelizedModels")

        count = 0
        if folderId != 0:
            children = vtk.vtkIdList()
            shNode.GetItemChildren(folderId, children)
            count = children.GetNumberOfIds()

        hasEnough = count >= 2
        self.ui.booleanCollapsibleButton.setEnabled(hasEnough)
        if hasEnough:
            self.ui.booleanCollapsibleButton.toolTip = ""
        else:
            self.ui.booleanCollapsibleButton.toolTip = (
                "Voxelize at least 2 models or segments first to enable boolean operations."
            )

    def _clearBooleanResult(self, *args):
        """Clear the result selector when inputs change."""
        self.ui.booleanOutputSelector.setCurrentNode(None)

    def onExportSectionToggled(self, collapsed):
        """Refresh the export model combo every time the section is expanded."""
        if not collapsed:
            self._populateExportCombo()

    def _populateBooleanCombos(self):
        """
        Fill booleanModelACombo and booleanModelBCombo with the names of all
        nodes inside the VoxelizedModels Subject Hierarchy folder.
        Only voxelized models are shown — not raw input models.
        """
        shNode  = slicer.mrmlScene.GetSubjectHierarchyNode()
        sceneId = shNode.GetSceneItemID()
        folderId = shNode.GetItemChildWithName(sceneId, "VoxelizedModels")

        names = []
        if folderId != 0:
            children = vtk.vtkIdList()
            shNode.GetItemChildren(folderId, children)
            for i in range(children.GetNumberOfIds()):
                node = shNode.GetItemDataNode(children.GetId(i))
                if node:
                    names.append(node.GetName())

        for combo in [self.ui.booleanModelACombo, self.ui.booleanModelBCombo]:
            current = combo.currentText
            combo.clear()
            for name in names:
                combo.addItem(name)
            # Restore previous selection if still available
            idx = combo.findText(current)
            if idx >= 0:
                combo.setCurrentIndex(idx)

    def _populateExportCombo(self, selectName=None):
        """
        Fill exportModelCombo with all models in the VoxelizedModels folder.
        If selectName is given, select that item automatically.
        """
        shNode   = slicer.mrmlScene.GetSubjectHierarchyNode()
        sceneId  = shNode.GetSceneItemID()
        folderId = shNode.GetItemChildWithName(sceneId, "VoxelizedModels")

        names = []
        if folderId != 0:
            children = vtk.vtkIdList()
            shNode.GetItemChildren(folderId, children)
            for i in range(children.GetNumberOfIds()):
                node = shNode.GetItemDataNode(children.GetId(i))
                if node:
                    names.append(node.GetName())

        current = selectName or self.ui.exportModelCombo.currentText
        self.ui.exportModelCombo.clear()
        for name in names:
            self.ui.exportModelCombo.addItem(name)

        idx = self.ui.exportModelCombo.findText(current)
        if idx >= 0:
            self.ui.exportModelCombo.setCurrentIndex(idx)

    def _populateMetricsModelCombo(self, selectName=None):
        """
        Fill metricsModelCombo with all models in the VoxelizedModels folder.
        Signals are blocked during population to prevent spurious table resets.
        """
        shNode   = slicer.mrmlScene.GetSubjectHierarchyNode()
        sceneId  = shNode.GetSceneItemID()
        folderId = shNode.GetItemChildWithName(sceneId, "VoxelizedModels")

        names = []
        if folderId != 0:
            children = vtk.vtkIdList()
            shNode.GetItemChildren(folderId, children)
            for i in range(children.GetNumberOfIds()):
                node = shNode.GetItemDataNode(children.GetId(i))
                if node:
                    names.append(node.GetName())

        current = selectName or self.ui.metricsModelCombo.currentText

        # Block signals so clear() and addItem() don't trigger _onMetricsModelChanged
        self.ui.metricsModelCombo.blockSignals(True)
        self.ui.metricsModelCombo.clear()
        for name in names:
            self.ui.metricsModelCombo.addItem(name)
        idx = self.ui.metricsModelCombo.findText(current)
        if idx >= 0:
            self.ui.metricsModelCombo.setCurrentIndex(idx)
        self.ui.metricsModelCombo.blockSignals(False)

        # Manually trigger table update for the selected model
        self._onMetricsModelChanged(self.ui.metricsModelCombo.currentIndex)

    def onBooleanApplyButton(self) -> None:
        """
        Perform the selected boolean operation on the two chosen voxelized
        models and place the result in the VoxelizedModels folder.
        """
        with slicer.util.tryWithErrorDisplay(_("Boolean operation failed."), waitCursor=True):
            nameA = self.ui.booleanModelACombo.currentText
            nameB = self.ui.booleanModelBCombo.currentText

            if not nameA or not nameB:
                raise ValueError("Please select two voxelized models.")

            # Allow same name only if they are genuinely different nodes
            nodeA = slicer.mrmlScene.GetFirstNodeByName(nameA)
            nodeB = slicer.mrmlScene.GetFirstNodeByName(nameB)

            if not nodeA or not nodeB:
                raise ValueError("Could not find the selected models in the scene.")
            if nodeA.GetID() == nodeB.GetID():
                raise ValueError("Model A and Model B must be different nodes.")

            # Estimate computation cost from total vertex count
            totalVerts = (nodeA.GetPolyData().GetNumberOfPoints() +
                          nodeB.GetPolyData().GetNumberOfPoints())

            # Warn if large — boolean re-voxelizes both models
            if totalVerts > 50000:
                msg = (
                    f"These models have {totalVerts:,} total vertices. "
                    f"This operation may take several minutes and will block the interface. Do you want to proceed?"
                )
                if not slicer.util.confirmOkCancelDisplay(msg, "Warning: Computationally intensive operation"):
                    return

            operation = self.ui.booleanOperationCombo.currentText
            if "Union" in operation:
                opKey    = "union"
                opSymbol = "U"
            elif "Intersection" in operation:
                opKey    = "intersection"
                opSymbol = "I"
            elif "B - A" in operation:
                opKey    = "difference_ba"
                opSymbol = "D_BA"
            else:
                opKey    = "difference"
                opSymbol = "D_AB"

            # Build unique output name e.g. "ModelA_U_ModelB"
            baseName   = f"{nameA}_{opSymbol}_{nameB}"

            # Use the node already in the selector if the user created/selected one
            existingNode = self.ui.booleanOutputSelector.currentNode()
            if existingNode:
                uniqueName = existingNode.GetName()
                # Delete existing node — logic will create a new one with correct data
                slicer.mrmlScene.RemoveNode(existingNode)
                self.ui.booleanOutputSelector.setCurrentNode(None)
            else:
                uniqueName = baseName
                counter    = 1
                while slicer.mrmlScene.GetFirstNodeByName(uniqueName):
                    uniqueName = f"{baseName}_{counter}"
                    counter   += 1

            resultNode = self.logic.applyBooleanOperation(nodeA, nodeB, opKey, uniqueName)
            self.logic.moveNodeToFolder(resultNode, "VoxelizedModels")

            # Show result in the boolean output selector
            self.ui.booleanOutputSelector.setCurrentNode(resultNode)

            # Refresh export combo so result appears there
            self._populateExportCombo(selectName=uniqueName)
            self._populateMetricsModelCombo(selectName=uniqueName)
            self._updateBooleanSectionState()

            self.setInfoLabel(f"Boolean result: {uniqueName}")

    # ------------------------------------------------------------------
    # Metrics table
    # ------------------------------------------------------------------

    # Row definitions: (row index, label text, internal key)
    _METRIC_ROWS = [
        (0, "Voxel count",                  "voxelCount"),
        (1, "Volume original [cm³]",        "volOriginal"),
        (2, "Volume voxelized [cm³]",       "volVoxelized"),
        (3, "ΔV [cm³]",                     "deltaVCm3"),
        (4, "ΔV [%]",                       "deltaVPct"),
        (5, "Excluded Mean ± Std",          "meanStd"),
        (6, "Excluded Median [IQR 5%-95%]", "medianIqr"),
    ]

    def _initMetricsTable(self):
        """Populate the table with row labels and N/A values."""
        import qt
        t = self.ui.metricsTable
        t.setRowCount(len(self._METRIC_ROWS))
        for row, label, _ in self._METRIC_ROWS:
            t.setItem(row, 0, qt.QTableWidgetItem(label))
            t.setItem(row, 1, qt.QTableWidgetItem("N/A"))
        t.horizontalHeader().setStretchLastSection(True)
        t.verticalHeader().setVisible(False)
        t.setEditTriggers(t.NoEditTriggers)
        t.viewport().update()

    def _onMetricsModelChanged(self, index):
        """Reload the table when the user picks a different model."""
        name = self.ui.metricsModelCombo.currentText
        if name and name in self._metricsStore:
            self._updateMetricsTable(self._metricsStore[name])
        else:
            self._initMetricsTable()

    def _initMetricsTable(self):
        """Populate the table with row labels and N/A values."""
        import qt
        t = self.ui.metricsTable
        t.setRowCount(len(self._METRIC_ROWS))
        for row, label, _ in self._METRIC_ROWS:
            t.setItem(row, 0, qt.QTableWidgetItem(label))
            t.setItem(row, 1, qt.QTableWidgetItem("N/A"))
        t.horizontalHeader().setStretchLastSection(True)
        t.verticalHeader().setVisible(False)
        t.setEditTriggers(t.NoEditTriggers)
        t.viewport().update()

    def _updateMetricsTable(self, values: dict):
        """
        Update the value column of the metrics table.
        """
        import qt
        t = self.ui.metricsTable
        t.setRowCount(len(self._METRIC_ROWS))
        for row, label, key in self._METRIC_ROWS:
            t.setItem(row, 0, qt.QTableWidgetItem(label))
            t.setItem(row, 1, qt.QTableWidgetItem(str(values.get(key, "N/A"))))
        t.viewport().update()
        slicer.app.processEvents()

    def onExportMetricsButton(self) -> None:
        """Export the stored metrics for the selected model to a CSV file."""
        import csv
        from qt import QFileDialog

        modelName = self.ui.metricsModelCombo.currentText
        if not modelName:
            slicer.util.errorDisplay("Please select a model first.")
            return

        # Use stored metrics for the selected model
        if modelName not in self._metricsStore:
            slicer.util.errorDisplay(f"No metrics stored for '{modelName}'. Run voxelization first.")
            return

        stored = self._metricsStore[modelName]

        defaultName = f"{modelName}_metrics.csv"
        startDir = str(self.ui.DirectoryButton.directory).strip()
        if not startDir or not os.path.isdir(startDir):
            startDir = os.path.expanduser("~")

        filePath = QFileDialog.getSaveFileName(
            None,
            "Save Metrics as CSV",
            os.path.join(startDir, defaultName),
            "CSV files (*.csv)"
        )
        if isinstance(filePath, tuple):
            filePath = filePath[0]
        if not filePath:
            return

        # Use _METRIC_ROWS to map keys to human-readable labels
        with open(filePath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Model", modelName])
            writer.writerow(["Metric", "Value"])
            for _, label, key in self._METRIC_ROWS:
                writer.writerow([label, stored.get(key, "N/A")])

        self.setInfoLabel(f"Metrics saved to: {filePath}")

    # ------------------------------------------------------------------
    # Export button
    # ------------------------------------------------------------------

    def onExportButton(self) -> None:
        self.setInfoLabel("")

        # Get the model chosen in the export combo
        exportName  = self.ui.exportModelCombo.currentText
        outputModel = slicer.mrmlScene.GetFirstNodeByName(exportName) if exportName else None

        if not outputModel:
            slicer.util.errorDisplay("Please select a model to export.")
            return

        directory = str(self.ui.DirectoryButton.directory).strip()

        if not directory:
            slicer.util.errorDisplay("Please select a valid export folder before exporting.")
            return

        baseFileName = outputModel.GetName()  # Ensure target directory exists; creates it if necessary

        # All voxelized models go directly into the chosen folder.
        os.makedirs(directory, exist_ok=True)

        paths = {
            "VTK": os.path.join(directory, f"{baseFileName}.vtk"),
            "STL": os.path.join(directory, f"{baseFileName}.stl"),
            "OBJ": os.path.join(directory, f"{baseFileName}.obj"),
            "MSH": os.path.join(directory, f"{baseFileName}.msh"),
            "PLY": os.path.join(directory, f"{baseFileName}.ply"),
        }

        selectedFormat = self.ui.exportFormatCombo.currentText
        if selectedFormat not in paths:
            slicer.util.errorDisplay(f"Unknown export format: {selectedFormat}")
            return

        with slicer.util.tryWithErrorDisplay(_("Export failed."), waitCursor=True):
            export_map = {
                "VTK": self.logic.exportModelVTK,
                "STL": self.logic.exportModelSTL,
                "OBJ": self.logic.exportModelOBJ,
                "MSH": self.logic.exportModelMSH,
                "PLY": self.logic.exportModelPLY,
            }
            export_map[selectedFormat](outputModel, paths[selectedFormat])
            self.setInfoLabel(f"Model saved to: {paths[selectedFormat]}")


#
# VoxelizationLogic
#


class VoxelizationLogic(ScriptedLoadableModuleLogic):

    def __init__(self) -> None:
        ScriptedLoadableModuleLogic.__init__(self)

    def getParameterNode(self):
        return VoxelizationParameterNode(super().getParameterNode())

    def _initModelNode(self, name: str):
        """
        Create a vtkMRMLModelNode using Slicer's standard scene infrastructure.
        Uses AddNewNodeByClass (Slicer's recommended way to add nodes) and
        CreateDefaultDisplayNodes (Slicer's standard display initialization).
        """
        node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", name)
        node.CreateDefaultDisplayNodes()
        dn = node.GetDisplayNode()
        if dn:
            dn.SetVisibility(True)
            dn.SetOpacity(1.0)
            dn.SetRepresentation(dn.SurfaceRepresentation)
        node.SetAttribute("Terminologies.TerminologyEntry", "")
        return node

    # ------------------------------------------------------------------
    # Core voxelization (unchanged from original)
    # ------------------------------------------------------------------

    def voxelizeModelToModel(self,
                             inputVolume: vtkMRMLScalarVolumeNode,
                             inputModel: vtkMRMLModelNode,
                             outputModel: vtkMRMLModelNode,
                             pitch: float,
                             threshold: float,
                             ui=None) -> None:

        from VoxelizationLib.logicUtils import (
            rasterizeModelToVolume, getVoxelizedModel,
            displayVoxelizedModel, computeMetrics, computeIntensityStats,
        )
        from numpy import count_nonzero

        if not inputVolume:
            raise ValueError("Invalid input volume")
        if not inputModel:
            raise ValueError("Invalid input model")
        if not outputModel:
            raise ValueError("Invalid output model")

        mask_orig          = rasterizeModelToVolume(inputModel, inputVolume)  # Step 1: rasterize original mesh to count occupied voxels
        originalVoxelCount = int(count_nonzero(mask_orig))
        grid_original      = mask_orig != 0  # Boolean occupancy grid for metric computation

        # Step 2: build the voxelized mesh; also returns excluded voxel occupancy percentages
        voxelizedModel, excludedOccupancy = getVoxelizedModel(inputModel, pitch, outputModel, threshold)
        displayVoxelizedModel(voxelizedModel)  # Attach a display node so the result appears in the 3D view

        mask                 = rasterizeModelToVolume(voxelizedModel, inputVolume)  # Step 3: rasterize the voxelized mesh for comparison
        voxelizedVoxelCount  = int(count_nonzero(mask))
        grid_voxelized       = mask != 0

        if ui:  # Step 5: compute and return metric values
            from VoxelizationLib.logicUtils import computeVolumeCm3, computeIntensityStats
            originalVolCm3  = computeVolumeCm3(inputModel)
            voxelizedVolCm3 = computeVolumeCm3(voxelizedModel)

            metrics = computeMetrics(grid_original, grid_voxelized, originalVoxelCount, voxelizedVoxelCount, originalVolCm3, voxelizedVolCm3)
            stats   = computeIntensityStats(excludedOccupancy)

            metricsValues = {
                "volOriginal":  f"{originalVolCm3:.4f}",
                "volVoxelized": f"{voxelizedVolCm3:.4f}",
                "deltaVCm3":    f"{metrics['deltaVCm3']:.4f}",
                "deltaVPct":    f"{metrics['deltaV']:.4f}",
                "voxelCount":   f"{int(voxelizedVoxelCount)}",
                "meanStd":      f"{stats['mean']:.2f}% \u00b1 {stats['std']:.2f}%",
                "medianIqr":    f"{stats['median']:.2f}% [{stats['p5']:.2f}% \u2013 {stats['p95']:.2f}%]",
            }

            slicer.app.processEvents()
            return metricsValues

    def segmentationToModel(self, segmentationNode: vtkMRMLSegmentationNode, segmentId: str):
        """
        Export exactly one segment (identified by segmentId) from
        segmentationNode into a temporary vtkMRMLModelNode and return it.

        Supports Slicer 5.x (ExportSingleSegmentToModelNode) and older
        versions via ExportVisibleSegmentsToModels.
        """
        tempModel = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", "TempSegmentModel")
        tempModel.CreateDefaultDisplayNodes()
        segLogic  = slicer.modules.segmentations.logic()

        # ---- Modern Slicer 5.x API ----
        if hasattr(segLogic, "ExportSingleSegmentToModelNode"):
            segLogic.ExportSingleSegmentToModelNode(segmentationNode, segmentId, tempModel)
            return tempModel

        # ---- Fallback: hide all segments except the chosen one ----
        segmentation = segmentationNode.GetSegmentation()
        displayNode  = segmentationNode.GetDisplayNode()

        visibilityMap = {}
        for i in range(segmentation.GetNumberOfSegments()):  # Save per-segment visibility so we can restore it afterward
            sid = segmentation.GetNthSegmentID(i)
            visibilityMap[sid] = displayNode.GetSegmentVisibility(sid)
            displayNode.SetSegmentVisibility(sid, sid == segmentId)  # Show only the desired segment, hide all others temporarily

        shNode       = slicer.mrmlScene.GetSubjectHierarchyNode()
        folderItemId = shNode.CreateFolderItem(shNode.GetSceneItemID(), "TempSegModels")  # Temporary SH folder captures the exported model nodes

        try:
            segLogic.ExportVisibleSegmentsToModels(segmentationNode, folderItemId)

            children = vtk.vtkIdList()
            shNode.GetItemChildren(folderItemId, children)

            if children.GetNumberOfIds() == 0:
                raise RuntimeError("No model was created from the selected segment.")

            exportedNode = shNode.GetItemDataNode(children.GetId(0))  # Copy polydata into our persistent node
            tempModel.SetAndObservePolyData(exportedNode.GetPolyData())
            slicer.mrmlScene.RemoveNode(exportedNode)  # Remove the intermediate exported node

        finally:
            # Restore visibility even if export raised an exception  # Always restore visibility - even when an exception was raised
            for sid, vis in visibilityMap.items():
                displayNode.SetSegmentVisibility(sid, vis)
            shNode.RemoveItem(folderItemId)  # Clean up the temporary Subject Hierarchy folder

        return tempModel

    # ------------------------------------------------------------------
    # Export helpers  (self was missing on STL/MSH in the original — fixed)
    # ------------------------------------------------------------------

    def exportModelPLY(self, modelNode, filePath):
        """Export modelNode as a PLY binary file (.ply)."""
        from VoxelizationLib.logicUtils import exportModelPLY
        exportModelPLY(modelNode, filePath)

    def exportModelVTK(self, modelNode, filePath):
        from VoxelizationLib.logicUtils import exportModelVTK
        exportModelVTK(modelNode, filePath)

    def exportModelSTL(self, modelNode, filePath):
        from VoxelizationLib.logicUtils import exportModelSTL
        exportModelSTL(modelNode, filePath)

    def exportModelOBJ(self, modelNode, filePath):
        from VoxelizationLib.logicUtils import exportModelOBJ
        exportModelOBJ(modelNode, filePath)

    def exportModelMSH(self, modelNode, filePath):
        """Export modelNode as a Gmsh MSH 2.2 file (.msh)."""
        from VoxelizationLib.logicUtils import exportModelMSH
        exportModelMSH(modelNode, filePath)

    # ------------------------------------------------------------------
    # Subject Hierarchy folder helper
    # ------------------------------------------------------------------

    def moveNodeToFolder(self, modelNode, folderName: str):
        """
        Move modelNode into a Subject Hierarchy folder called folderName.
        The folder is created once and reused on every subsequent call,
        so all voxelized results are grouped together in one place.
        """
        shNode   = slicer.mrmlScene.GetSubjectHierarchyNode()
        sceneId  = shNode.GetSceneItemID()

        folderId = shNode.GetItemChildWithName(sceneId, folderName)
        if folderId == 0:
            folderId = shNode.CreateFolderItem(sceneId, folderName)

        nodeItemId = shNode.GetItemByDataNode(modelNode)
        shNode.SetItemParent(nodeItemId, folderId)

    # ------------------------------------------------------------------
    # Boolean operations
    # ------------------------------------------------------------------

    def applyBooleanOperation(self, nodeA, nodeB, operation: str, outputName: str):
        """
        Boolean operation between two voxelized hex mesh models.

        Following the reviewer's suggestion: a hex mesh on a uniform grid is
        equivalent to a labelmap image. We resample both models onto a common
        grid as dense boolean numpy arrays (the labelmap-equivalent
        representation), then perform the boolean operation with a single
        numpy call -- exactly as suggested for labelmap volumes.

        Pipeline
        --------
        1. Extract voxel centers from each hex mesh (face-normal method).
        2. Define a common grid (origin, pitch) covering both bounding boxes.
        3. Rasterize each model's centers into a dense boolean array on that
           common grid -- this dense array IS the labelmap-equivalent.
        4. Boolean operation = one numpy call (logical_or / logical_and / etc).
        5. Rebuild the hex surface from the resulting dense array.
        """
        import trimesh
        import numpy as np
        from numpy import hstack, full, int64

        def extractCentersAndPitch(polyData):
            """
            Extract voxel centers from a hex mesh polydata using face
            centers and normals:  center = face_center - normal * pitch/2
            Pitch is estimated from vertex spacing (adjacent cubes share
            corners, so min diff between unique vertex coords = pitch).
            """
            pts   = vtk.util.numpy_support.vtk_to_numpy(polyData.GetPoints().GetData()).astype(np.float64)
            cells = vtk.util.numpy_support.vtk_to_numpy(polyData.GetPolys().GetData())
            tris  = cells.reshape(-1, 4)[:, 1:]

            v0 = pts[tris[:, 0]]
            v1 = pts[tris[:, 1]]
            v2 = pts[tris[:, 2]]

            face_min     = np.minimum(np.minimum(v0, v1), v2)
            face_max     = np.maximum(np.maximum(v0, v1), v2)
            face_centers = (face_min + face_max) / 2.0

            uniqueXv = np.unique(np.round(pts[:, 0], 4))
            diffsv   = np.diff(uniqueXv)
            diffsv   = diffsv[diffsv > 1e-4]
            pitch    = float(np.min(diffsv)) if len(diffsv) > 0 else 1.0

            normals = np.cross(v1 - v0, v2 - v0)
            norms   = np.linalg.norm(normals, axis=1, keepdims=True)
            norms   = np.where(norms < 1e-10, 1.0, norms)
            normals = normals / norms
            snapped = np.zeros_like(normals)
            axisIdx = np.argmax(np.abs(normals), axis=1)
            for i, ax in enumerate(axisIdx):
                snapped[i, ax] = np.sign(normals[i, ax])

            centers = face_centers - snapped * (pitch / 2.0)
            centers = np.round(centers, 4)
            centers = np.unique(centers, axis=0)
            return centers, pitch

        centersA, pitchA = extractCentersAndPitch(nodeA.GetPolyData())
        centersB, pitchB = extractCentersAndPitch(nodeB.GetPolyData())

        pitch = min(pitchA, pitchB)

        if abs(pitchA - pitchB) > pitch * 0.01:
            slicer.util.infoDisplay(
                f"Model A edge size ({pitchA:.2f} mm) and Model B edge size ({pitchB:.2f} mm) differ. "
                f"Boolean operation will be performed at the smaller edge size ({pitch:.2f} mm).",
                windowTitle="Edge size mismatch"
            )

        # --------------------------------------------------------------
        # Step 2 -- common grid covering both models
        # --------------------------------------------------------------
        allCenters = np.vstack([centersA, centersB])
        gridOrigin = np.floor(allCenters.min(axis=0) / pitch) * pitch
        gridMax    = np.ceil(allCenters.max(axis=0) / pitch) * pitch
        gridDims   = np.round((gridMax - gridOrigin) / pitch).astype(int) + 1

        # --------------------------------------------------------------
        # Step 3 -- rasterize each model onto the common grid as a dense
        # boolean array. This dense array is the direct equivalent of a
        # labelmap image / uniform-grid hex mesh, as the reviewer noted.
        # --------------------------------------------------------------
        def toDenseArray(centers, origin, pitch, dims):
            idx   = np.floor((centers - origin) / pitch + 0.5).astype(int)
            idx   = np.clip(idx, 0, dims - 1)
            array = np.zeros(tuple(dims), dtype=bool)
            array[idx[:, 0], idx[:, 1], idx[:, 2]] = True
            return array

        labelmapA = toDenseArray(centersA, gridOrigin, pitch, gridDims)
        labelmapB = toDenseArray(centersB, gridOrigin, pitch, gridDims)

        # --------------------------------------------------------------
        # Step 4 -- boolean operation: a single numpy call, exactly as
        # suggested for labelmap volumes.
        # --------------------------------------------------------------
        if operation == "union":
            resultArray = np.logical_or(labelmapA, labelmapB)
        elif operation == "intersection":
            resultArray = np.logical_and(labelmapA, labelmapB)
        elif operation == "difference":
            resultArray = np.logical_and(labelmapA, np.logical_not(labelmapB))
        elif operation == "difference_ba":
            resultArray = np.logical_and(labelmapB, np.logical_not(labelmapA))
        else:
            raise ValueError(f"Unknown operation: {operation}")

        if not resultArray.any():
            raise ValueError(
                "Boolean operation produced an empty result. "
                "For Intersection/Difference make sure the models overlap."
            )

        # --------------------------------------------------------------
        # Step 5 -- rebuild hex surface from the resulting dense array
        # --------------------------------------------------------------
        resultIndices = np.argwhere(resultArray)
        localOffset   = resultIndices.min(axis=0)
        localIdx      = resultIndices - localOffset
        dims          = localIdx.max(axis=0) + 1

        denseMatrix = np.zeros(dims, dtype=bool)
        denseMatrix[localIdx[:, 0], localIdx[:, 1], localIdx[:, 2]] = True

        worldOrigin    = gridOrigin + localOffset.astype(float) * pitch
        transform      = np.eye(4) * pitch
        transform[0, 3] = worldOrigin[0]
        transform[1, 3] = worldOrigin[1]
        transform[2, 3] = worldOrigin[2]
        transform[3, 3] = 1.0

        resultGrid = trimesh.voxel.VoxelGrid(
            trimesh.voxel.encoding.DenseEncoding(denseMatrix),
            transform
        )
        surface = resultGrid.as_boxes()

        v_out        = surface.vertices
        f_out        = surface.faces
        out_poly     = vtk.vtkPolyData()
        v_vtk        = vtk.util.numpy_support.numpy_to_vtk(v_out, deep=True)
        pts_vtk      = vtk.vtkPoints()
        pts_vtk.SetData(v_vtk)
        out_poly.SetPoints(pts_vtk)
        num_faces    = f_out.shape[0]
        cells_array  = hstack([full((num_faces, 1), 3), f_out]).astype(int64)
        cells_vtk    = vtk.util.numpy_support.numpy_to_vtkIdTypeArray(cells_array.ravel(), deep=True)
        connectivity = vtk.vtkCellArray()
        connectivity.SetCells(num_faces, cells_vtk)
        out_poly.SetPolys(connectivity)

        resultNode = self._initModelNode(outputName)
        resultNode.SetAndObservePolyData(out_poly)

        return resultNode
