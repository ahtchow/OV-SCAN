#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "PATCHWORK::patchworkpp" for configuration "Release"
set_property(TARGET PATCHWORK::patchworkpp APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(PATCHWORK::patchworkpp PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libpatchworkpp.so"
  IMPORTED_SONAME_RELEASE "libpatchworkpp.so"
  )

list(APPEND _cmake_import_check_targets PATCHWORK::patchworkpp )
list(APPEND _cmake_import_check_files_for_PATCHWORK::patchworkpp "${_IMPORT_PREFIX}/lib/libpatchworkpp.so" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
