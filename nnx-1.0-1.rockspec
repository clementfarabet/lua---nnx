
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
   "sys",
   "xlua"
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

         find_package (OpenMP REQUIRED)

         if (OPENMP_FOUND)
             message (STATUS "OpenMP Found with compiler flag : ${OpenMP_C_FLAGS}")
             set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
         endif (OPENMP_FOUND)

         set (CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

	include_directories (${TORCH_INCLUDE_DIR} ${PROJECT_SOURCE_DIR})

	 SET(WITH_CUDA OFF CACHE BOOL "Compile qlua and associated packages")

	IF(EXISTS "${TORCH_BIN_DIR}/../include/THC/THC.h")
	FIND_PACKAGE(CUDA 4.0)
	
	IF (CUDA_FOUND)
	   # bug on Apple
	   IF(APPLE)
	     LINK_DIRECTORIES("/usr/local/cuda/lib/")
	   ENDIF(APPLE)

	find_library (TORCH_THC THC ${TORCH_BIN_DIR}/../lib NO_DEFAULT_PATH)

	
	set (TORCH_LIBRARIES  ${TORCH_TH} ${TORCH_THC} ${TORCH_luaT} ${TORCH_lua})
	set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DWITH_CUDA")

         cuda_add_library (lbfgs SHARED lbfgs.c)
         target_link_libraries (lbfgs ${TORCH_LIBRARIES})
	 CUDA_ADD_CUBLAS_TO_TARGET(lbfgs)

	ELSE (CUDA_FOUND)

	    MESSAGE(STATUS "Disabling CUDA (CUDA 4.0 required, and not found)")
	    MESSAGE(STATUS "If CUDA 4.0 is installed, then specify CUDA_TOOLKIT_ROOT_DIR")

	    set (TORCH_LIBRARIES ${TORCH_TH} ${TORCH_luaT} ${TORCH_lua})

         add_library (lbfgs SHARED lbfgs.c)
         target_link_libraries (lbfgs ${TORCH_LIBRARIES})

	ENDIF (CUDA_FOUND)
	ENDIF(EXISTS "${TORCH_BIN_DIR}/../include/THC/THC.h")

         install_targets (/lib lbfgs)
 
	

         include_directories (${TORCH_INCLUDE_DIR} ${PROJECT_SOURCE_DIR})
         add_library (nnx SHARED init.c)
         link_directories (${TORCH_LIBRARY_DIR})
         target_link_libraries (nnx ${TORCH_LIBRARIES})

         install_files(/lua/nnx init.lua)
         install_files(/lua/nnx Abs.lua)
         install_files(/lua/nnx ConfusionMatrix.lua)
         install_files(/lua/nnx DistNLLCriterion.lua)
         install_files(/lua/nnx Logger.lua)
         install_files(/lua/nnx Probe.lua)
         install_files(/lua/nnx HardShrink.lua)
         install_files(/lua/nnx Narrow.lua)
         install_files(/lua/nnx Power.lua)
         install_files(/lua/nnx Square.lua)
         install_files(/lua/nnx Sqrt.lua)
         install_files(/lua/nnx Threshold.lua)
         install_files(/lua/nnx OmpModule.lua)
         install_files(/lua/nnx SpatialClassifier.lua)
         install_files(/lua/nnx SpatialMaxPooling.lua)
         install_files(/lua/nnx SpatialLinear.lua)
         install_files(/lua/nnx SpatialPadding.lua)
         install_files(/lua/nnx SpatialNormalization.lua)
         install_files(/lua/nnx SpatialUpSampling.lua)
         install_files(/lua/nnx SpatialReSampling.lua)
         install_files(/lua/nnx SpatialConvolutionSparse.lua)
         install_files(/lua/nnx SuperCriterion.lua)
         install_files(/lua/nnx SpatialCriterion.lua)
         install_files(/lua/nnx Trainer.lua)
         install_files(/lua/nnx OnlineTrainer.lua)
         install_files(/lua/nnx DataSet.lua)
         install_files(/lua/nnx DataList.lua)
         install_files(/lua/nnx DataSetLabelMe.lua)
         install_files(/lua/nnx CMulTable.lua)
         install_files(/lua/nnx CAddTable.lua)
         install_files(/lua/nnx CDivTable.lua)
         install_files(/lua/nnx CSubTable.lua)
         install_files(/lua/nnx Replicate.lua)
         install_files(/lua/nnx SpatialFovea.lua)
         install_files(/lua/nnx SpatialMSECriterion.lua)
         install_files(/lua/nnx SpatialClassNLLCriterion.lua)
         install_files(/lua/nnx SparseCriterion.lua)
         install_files(/lua/nnx SpatialSparseCriterion.lua)
         install_files(/lua/nnx SpatialGraph.lua)
         install_files(/lua/nnx SpatialColorTransform.lua)
         install_files(/lua/nnx SpatialRecursiveFovea.lua)
         install_files(/lua/nnx Optimization.lua)
         install_files(/lua/nnx LBFGSOptimization.lua)
         install_files(/lua/nnx SGDOptimization.lua)
         install_files(/lua/nnx BatchOptimization.lua)
         install_files(/lua/nnx BatchTrainer.lua)
         add_subdirectory (test)
         install_targets(/lib nnx)
   ]],

   variables = {
      CMAKE_INSTALL_PREFIX = "$(PREFIX)"
   }
}
