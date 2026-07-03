cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(Voxelization)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://github.com/DomenicoRiggio/SlicerModelsVoxelization")
set(EXTENSION_CATEGORY "Utilities")
set(EXTENSION_CONTRIBUTORS "Kimia Ghodousipour (Institute of Biomedical Engineering, KIT), Laura Lichtlein (Institute of Biomedical Engineering, KIT), Ciro Benito Raggio (Institute of Biomedical Engineering, KIT), Domenico Riggio (Institute of Biomedical Engineering, KIT), ")
set(EXTENSION_DESCRIPTION "This extensions aims at voxelizing meshes.")
set(EXTENSION_ICONURL "https://raw.githubusercontent.com/DomenicoRiggio/SlicerModelsVoxelization/main/Voxelization/Resources/Icons/Voxelization.png")
set(EXTENSION_SCREENSHOTURLS "https://raw.githubusercontent.com/DomenicoRiggio/SlicerModelsVoxelization/main/Voxelization/Resources/Icons/Examples.png")
set(EXTENSION_DEPENDS "NA") # Specified as a list or "NA" if no dependencies

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(Voxelization)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
