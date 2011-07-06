#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/Template.c"
#else

static int nn_(Template_forward)(lua_State *L)
{

}

static int nn_(Template_backward)(lua_State *L)
{

}

static const struct luaL_Reg nn_(Template__) [] = {
  {"Template_forward", nn_(Template_forward)},
  {"Template_backward", nn_(Template_backward)},
  {NULL, NULL}
};

static void nn_(Template_init)(lua_State *L)
{
  luaT_pushmetaclass(L, torch_(Tensor_id));
  luaT_registeratname(L, nn_(Template__), "nn");
  lua_pop(L,1);
}

#endif
