
package = "nnx"
version = "1.0-1"

source = {
   url = "nnx-1.0-1.tgz"
}

description = {
   summary = "An extension to Torch7's nn package.",
   detailed = [[
         This package provides extra trainable modules,
         which naturally extend the nn package. 
         Some of those might get marged into the original
         nn package, at some point. For this reason,
         all the modules from nnx are appended to nn.
   ]],
   homepage = "",
   license = "MIT/X11" -- or whatever you like
}

dependencies = {
   "lua >= 5.1",
   "torch",
   "xlua",
   "lunit"
}

build = {
   type = "cmake",

   cmake = [[
         cmake_minimum_required(VERSION 2.8)

         set (CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR})

         # infer path for Torch7
         string (REGEX REPLACE "(.*)lib/luarocks/rocks.*" "\\1" TORCH_PREFIX "${CMAKE_INSTALL_PREFIX}" )
         message (STATUS "Found Torch7, installed in: " ${TORCH_PREFIX})

         find_package (Torch REQUIRED)

         set (CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

         include_directories (${TORCH_INCLUDE_DIR} ${PROJECT_SOURCE_DIR})
         add_library (nnx SHARED init.c)
         link_directories (${TORCH_LIBRARY_DIR})
         target_link_libraries (nnx ${TORCH_LIBRARIES})

         install_files(/lua/nnx init.lua)
         install_files(/lua/nnx Abs.lua)
         install_files(/lua/nnx ConfusionMatrix.lua)
         install_files(/lua/nnx HardShrink.lua)
         install_files(/lua/nnx Narrow.lua)
         install_files(/lua/nnx Power.lua)
         install_files(/lua/nnx Square.lua)
         install_files(/lua/nnx Sqrt.lua)
         install_files(/lua/nnx Threshold.lua)
         install_files(/lua/nnx SpatialLinear.lua)
         add_subdirectory (test)
         install_targets(/lib nnx)
   ]],

   variables = {
      CMAKE_INSTALL_PREFIX = "$(PREFIX)"
   }
}
