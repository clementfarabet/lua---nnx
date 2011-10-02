
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

   variables = {
      CMAKE_INSTALL_PREFIX = "$(PREFIX)"
   }
}
