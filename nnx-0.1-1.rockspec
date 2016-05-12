package = "nnx"
version = "0.1-1"

source = {
   url = "git://github.com/clementfarabet/lua---nnx",
   tag = "master"
}

description = {
   summary = "A completely unstable and experimental package that extends Torch's builtin nn library",
   detailed = [[
This is an experimental package that extends nn. You've be warned!
   ]],
   homepage = "https://github.com/clementfarabet/lua---nnx",
   license = "BSD"
}

dependencies = {
   "torch >= 7.0",
   "xlua >= 1.0"
}

build = {
   type = "command",
   build_command = [[
cmake -E make_directory build && cd build && cmake .. -DLUALIB=$(LUALIB)  -DLUA_INCDIR="$(LUA_INCDIR)" -DLUA_LIBDIR="$(LUA_LIBDIR)" -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$(LUAROCKS_PREFIX)" -DCMAKE_INSTALL_PREFIX="$(PREFIX)" && $(MAKE)
   ]],
   install_command = "cd build && $(MAKE) install"
}
