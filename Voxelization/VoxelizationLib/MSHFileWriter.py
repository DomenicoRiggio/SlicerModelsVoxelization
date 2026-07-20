import vtk
import slicer


class MSHFileWriter:
    """
    Slicer scripted file writer plugin for Gmsh MSH 2.2 format.

    Registering this class with slicer.app.ioManager() makes ".msh" appear
    as a selectable format directly in Slicer's standard "Save Data" dialog
    for any vtkMRMLModelNode -- no custom export GUI needed.

    Follows the pattern from:
    Applications/SlicerApp/Testing/Python/SlicerScriptedFileReaderWriterTest.py
    """

    def __init__(self, parent):
        self.parent = parent

    def description(self):
        return "Gmsh Mesh (MSH 2.2)"

    def fileType(self):
        return "MSHFile"

    def extensions(self, obj):
        return ["Gmsh Mesh (*.msh)"]

    def canWriteObjectConfidence(self, obj):
        # Only offer this writer for model nodes with polydata geometry
        if not obj.IsA("vtkMRMLModelNode"):
            return 0.0
        if not obj.GetPolyData() or obj.GetPolyData().GetNumberOfPoints() == 0:
            return 0.0
        # Default confidence, lets the user choose it from the format dropdown
        return 0.5

    def write(self, properties):
        try:
            node     = slicer.mrmlScene.GetNodeByID(properties["nodeID"])
            filePath = properties["fileName"]

            from VoxelizationLib.logicUtils import exportModelMSH
            exportModelMSH(node, filePath)

        except Exception as e:
            import traceback
            traceback.print_exc()
            errorMessage = f"Failed to write MSH file: {str(e)}"
            self.parent.userMessages().AddMessage(vtk.vtkCommand.ErrorEvent, errorMessage)
            return False

        self.parent.writtenNodes = [node.GetID()]
        return True
