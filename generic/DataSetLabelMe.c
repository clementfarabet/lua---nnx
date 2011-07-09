#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/DataSetLabelMe.c"
#else

static int nn_(DataSetLabelMe_extract)(lua_State *L)
{
  const void* torch_ShortStorage_id = luaT_checktypename2id(L, "torch.ShortStorage");
  int tags = 1;
  THTensor *mask = luaT_checkudata(L, 2, torch_(Tensor_id));
  int x_start = lua_tonumber(L, 3);
  int x_end = lua_tonumber(L, 4);
  int y_start = lua_tonumber(L, 5);
  int y_end = lua_tonumber(L, 6);
  int idx = lua_tonumber(L, 7);

  int x,y,label,tag,size;
  THShortStorage *data;
  for (x=x_start; x<=x_end; x++) {
    for (y=y_start; y<=y_end; y++) {
      label = THTensor_(get2d)(mask, x-1, y-1);                                   // label = mask[x][y]
      lua_rawgeti(L, tags, label);                                              // tag = tags[label]
      tag = lua_gettop(L);
      lua_pushstring(L, "size"); lua_rawget(L, tag);                            // size = tag.size
      size = lua_tonumber(L,-1); lua_pop(L,1);
      lua_pushstring(L, "size"); lua_pushnumber(L, size+3); lua_rawset(L, tag); // tag.size = size + 3
      lua_pushstring(L, "data"); lua_rawget(L, tag);                            // data = tag.data
      data = luaT_checkudata(L, -1, torch_ShortStorage_id); lua_pop(L, 1);
      data->data[size] = x;                                                     // data[size+1] = x
      data->data[size+1] = y;                                                   // data[size+1] = y
      data->data[size+2] = idx;                                                 // data[size+1] = idx
      lua_pop(L, 1);
    }
  }
  return 0;
}

static int nn_(DataSetLabelMe_backward)(lua_State *L)
{

}

static const struct luaL_Reg nn_(DataSetLabelMe__) [] = {
  {"DataSetLabelMe_extract", nn_(DataSetLabelMe_extract)},
  {NULL, NULL}
};

static void nn_(DataSetLabelMe_init)(lua_State *L)
{
  luaT_pushmetaclass(L, torch_(Tensor_id));
  luaT_registeratname(L, nn_(DataSetLabelMe__), "nn");
  lua_pop(L,1);
}

#endif
