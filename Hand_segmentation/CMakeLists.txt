# CMakeList.txt: progetto CMake per Hand_segmentation. Includere l'origine e definire
# qui la logica specifica del progetto.
#

cmake_minimum_required (VERSION 3.8)

project ("Hand_segmentation")
set(OpenCV_DIR "C:/Users/Ricca/workspace/opencv/build/")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(includes, "C:/Users/Ricca/Desktop/UNIPD/COMPUTER VISION/finalProject/Hand_segmentation/" )
# Aggiungere l'origine all'eseguibile di questo progetto.
add_executable (Hand_segmentation    "main.cpp" "Segmentation.h" "Segmentation.cpp")
target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS})

# TODO: Aggiungere i test e, se necessario, installare le destinazioni.

