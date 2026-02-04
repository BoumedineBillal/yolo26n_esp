set(srcs app_main.cpp)

set(requires esp-dl)

idf_build_get_property(component_targets __COMPONENT_TARGETS)
if ("___idf_espressif__esp-dl" IN_LIST component_targets)
   idf_component_get_property(espdl_dir espressif__esp-dl COMPONENT_DIR)
elseif("___idf_esp-dl" IN_LIST component_targets)
   idf_component_get_property(espdl_dir esp-dl COMPONENT_DIR)
endif()
set(cmake_dir ${espdl_dir}/fbs_loader/cmake)
include(${cmake_dir}/utilities.cmake)

# Embed model and test images
# --- Model Selection ---
# To use the 640x640 model: 
#   1. Change the line below to: models/yolo26n_640.espdl
#   2. Ensure CONFIG_YOLO_INPUT_WIDTH/HEIGHT in kconfig (or defines) is 640 in app_mainv3.cpp
#
# Default: 512x512
set(embed_files models/yolo26n_512.espdl
                images/bus.jpg
                images/person.jpg
)

idf_component_register(SRCS ${srcs}
                       REQUIRES ${requires}
                       EMBED_FILES ${embed_files})

target_add_aligned_binary_data(${COMPONENT_LIB} models/yolo26n_512.espdl BINARY)
