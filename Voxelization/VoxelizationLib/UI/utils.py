import os
import base64

CONTRIBUTORS = [
    "Laura Lichtlein (Karlsruhe Institute of Technology, Germany)",
    "Domenico Riggio (Karlsruhe Institute of Technology, Germany)",
    "Ciro Benito Raggio (Karlsruhe Institute of Technology, Germany)",
    "Kimia Ghodousipour (Karlsruhe Institute of Technology, Germany)",
]

MODULE_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
LOGO_PATH       = os.path.join(MODULE_ROOT_DIR, 'Resources', 'Icons', 'Voxelization.png').replace('\\', '/')

def _logo_html(width=200, height=200):
    """Return an <img> tag with the logo embedded as base64 (works in QTextBrowser)."""
    if os.path.exists(LOGO_PATH):
        with open(LOGO_PATH, 'rb') as f:
            b64 = base64.b64encode(f.read()).decode()
        return f'<img src="data:image/png;base64,{b64}" width="{width}" height="{height}">'
    return ''

def getHelpText():
    return f"""
<center>
    {_logo_html(200, 200)}
</center>
<br/>
<b>Description</b>
<br/>
SlicerModelsVoxelization is an open-source 3D Slicer extension for converting surface meshes
and segmentations into solid cubical voxel models, with support for boolean operations,
quantitative metrics, and multi-format export.
<br/><br/>
<b>Key Features</b>
<ul>
    <li>Voxelize model nodes or segmentations (per-segment selection)</li>
    <li>Adjustable pitch with user-defined maximum and occupancy threshold</li>
    <li>Boolean operations: Union, Intersection, Difference (A−B), Difference (B−A)</li>
    <li>Quantitative metrics: volume, ΔV, excluded voxel statistics with CSV export</li>
    <li>Export in VTK, STL, OBJ, MSH, PLY, OFF formats</li>
</ul>
<br/>
<b>How to Use</b>
<ul>
    <li>Load a volume and a model or segmentation into the scene</li>
    <li>Select <b>Input Type</b>: Segmentation or Model</li>
    <li>Select the input, reference volume, pitch [cm³] and threshold</li>
    <li>Click <b>Apply</b> to generate the voxelized model</li>
    <li>For boolean operations, voxelize at least two models first, then expand <b>Boolean Operations</b></li>
    <li>Select Model A, Model B and the operation, then click <b>Apply Boolean Operation</b></li>
    <li>Use the <b>Export</b> section to save results in your preferred format</li>
</ul>
<br/>
<b>Boolean Operations Note</b>
<ul>
    <li>Boolean operations work best when both models share the same pitch</li>
    <li>If pitches differ, the operation is performed at the smaller pitch and
        the coarser model is automatically re-sampled</li>
</ul>
<br/>
<b>More info</b>
<ul>
    <li>View the source code on GitHub:
        <a href="https://github.com/DomenicoRiggio/SlicerModelsVoxelization">
        https://github.com/DomenicoRiggio/SlicerModelsVoxelization</a></li>
</ul>
"""

# Keep HELP_TEXT as a string for backward compatibility
HELP_TEXT = getHelpText()
