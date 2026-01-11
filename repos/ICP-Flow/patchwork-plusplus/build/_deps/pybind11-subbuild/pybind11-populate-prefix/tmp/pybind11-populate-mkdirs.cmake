# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/OV-SCAN/repos/ICP-Flow/patchwork-plusplus/build/_deps/pybind11-src"
  "/OV-SCAN/repos/ICP-Flow/patchwork-plusplus/build/_deps/pybind11-build"
  "/OV-SCAN/repos/ICP-Flow/patchwork-plusplus/build/_deps/pybind11-subbuild/pybind11-populate-prefix"
  "/OV-SCAN/repos/ICP-Flow/patchwork-plusplus/build/_deps/pybind11-subbuild/pybind11-populate-prefix/tmp"
  "/OV-SCAN/repos/ICP-Flow/patchwork-plusplus/build/_deps/pybind11-subbuild/pybind11-populate-prefix/src/pybind11-populate-stamp"
  "/OV-SCAN/repos/ICP-Flow/patchwork-plusplus/build/_deps/pybind11-subbuild/pybind11-populate-prefix/src"
  "/OV-SCAN/repos/ICP-Flow/patchwork-plusplus/build/_deps/pybind11-subbuild/pybind11-populate-prefix/src/pybind11-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/OV-SCAN/repos/ICP-Flow/patchwork-plusplus/build/_deps/pybind11-subbuild/pybind11-populate-prefix/src/pybind11-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/OV-SCAN/repos/ICP-Flow/patchwork-plusplus/build/_deps/pybind11-subbuild/pybind11-populate-prefix/src/pybind11-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
