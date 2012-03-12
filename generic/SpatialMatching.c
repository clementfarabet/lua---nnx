#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/SpatialMatching.c"
#else

#define square(x) ((x)*(x))
#define max(x,y) (((x)>(y)) ? (x) : (y))
#define min(x,y) (((x)>(y)) ? (y) : (x))

static int nn_(SpatialMatching_updateOutput)(lua_State *L)
{
  // get all params
  THTensor *input1 = luaT_checkudata(L, 2, torch_(Tensor_id));
  THTensor *input2 = luaT_checkudata(L, 3, torch_(Tensor_id));
  int maxw = luaT_getfieldcheckint(L, 1, "maxw");
  int maxh = luaT_getfieldcheckint(L, 1, "maxh");
  int full_output = luaT_getfieldcheckboolean(L, 1, "full_output");
  THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_(Tensor_id));

  // dims
  int iwidth = input1->size[2];
  int iheight = input1->size[1];
  int ichannels = input1->size[0];

  // zero output
  THTensor_(fill)(output, 1e30);

  // get strides
  long *i1s = input1->stride;
  long *i2s = input2->stride;
  long *os  = output->stride;

  // get pointers
  real *input1_p = THTensor_(data)(input1);
  real *input2_p = THTensor_(data)(input2);
  real *output_p = THTensor_(data)(output);

  // compute output
  int x1,y1,x2,y2,k;
  if (full_output) {
    // get halves of window size
    int halfh1 = ceil((real)maxh/2)-1;
    int halfh2 = floor((real)maxh/2)+1;
    int halfw1 = ceil((real)maxw/2)-1;
    int halfw2 = floor((real)maxw/2)+1;

    //#pragma omp parallel for private(x1,y1,x2,y2,k)
    for (y1 = 0; y1 < iheight; y1++) {
      for (x1 = 0; x1 < iwidth; x1++) {
	for (y2 = max(0,y1-halfh1); y2 < min(iheight,y1+halfh2); y2++) {
	  for (x2 = max(0,(x1-halfw1)); x2 < min(iwidth,x1+halfw2); x2++) {
	    real dist = 0;
	    for (k=0; k<ichannels; k++) {
	      dist += square(input1_p[k*i1s[0] + y1*i1s[1] + x1*i1s[2]] - input2_p[k*i2s[0] + y2*i2s[1] + x2*i2s[2]]);
	      //dist += square(THTensor_(get3d)(input1, k, y1, x1) - THTensor_(get3d)(input2, k, y2, x2));
	    }
	    long dy = y2-y1 + halfh1;
	    long dx = x2-x1 + halfw1;
	    output_p[y1*os[0] + x1*os[1] + dy*os[2] + dx*os[3]] = dist;
	    //THTensor_(set4d)(output, y1, x1, dy, dx, dist);
	  }
	}
      }
    }
  } else {
    //#pragma omp parallel for private(x1,y1,x2,y2,k)
    for (y1 = 0; y1 < iheight; y1++) {
      for (x1 = 0; x1 < iwidth; x1++) {
	for (y2 = y1; y2 < y1+maxh; y2++) {
	  for (x2 = x1; x2 < x1+maxw; x2++) {
	    real dist = 0;
	    for (k = 0; k < ichannels; k++) {
	      dist += square(input1_p[k*i1s[0] + y1*i1s[1] + x1*i1s[2]] - input2_p[k*i2s[0] + y2*i2s[1] + x2*i2s[2]]);
	      //dist += square(THTensor_(get3d)(input1, k, y1, x1) - THTensor_(get3d)(input2, k, y2, x2));
	    }
	    output_p[y1*os[0] + x1*os[1] + (y2-y1)*os[2] + (x2-x1)*os[3]] = dist;
	    //THTensor_(set4d)(output, y1, x1, y2-y1, x2-x1, dist);
	  }
	}
      }
    }
  }
    

  // done
  return 1;
}

static int nn_(SpatialMatching_updateGradInput)(lua_State *L)
{
  // get all params
  THTensor *input1 = luaT_checkudata(L, 2, torch_(Tensor_id));
  THTensor *input2 = luaT_checkudata(L, 3, torch_(Tensor_id));
  THTensor *gradInput1 = luaT_getfieldcheckudata(L, 1, "gradInput1", torch_(Tensor_id));
  THTensor *gradInput2 = luaT_getfieldcheckudata(L, 1, "gradInput2", torch_(Tensor_id));
  THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_(Tensor_id));
  THTensor *gradOutput = luaT_checkudata(L, 4, torch_(Tensor_id));
  int full_output = luaT_getfieldcheckboolean(L, 1, "full_output");
  int maxw = luaT_getfieldcheckint(L, 1, "maxw");
  int maxh = luaT_getfieldcheckint(L, 1, "maxh");

  // dims
  int iwidth = input1->size[2];
  int iheight = input1->size[1];
  int ichannels = input1->size[0];

  // resize gradInput
  //THTensor_(zero)(gradInput1);
  //THTensor_(zero)(gradInput2);

  // get strides
  long *i1s = input1->stride;
  long *i2s = input2->stride;
  long *gi1s = gradInput1->stride;
  long *gi2s = gradInput2->stride;
  //long *os  = output->stride;
  long *gos = gradOutput->stride;
  
  // get pointers
  real *input1_p = THTensor_(data)(input1);
  real *input2_p = THTensor_(data)(input2);
  real *gradInput1_p = THTensor_(data)(gradInput1);
  real *gradInput2_p = THTensor_(data)(gradInput2);
  //real *output_p = THTensor_(data)(output);
  real *gradOutput_p = THTensor_(data)(gradOutput);
  
  // compute gradients
  int x1, y1, x2, y2, k;
  if (full_output) {
    // get halves of window size
    int halfh1 = ceil((real)maxh/2)-1;
    int halfh2 = floor((real)maxh/2)+1;
    int halfw1 = ceil((real)maxw/2)-1;
    int halfw2 = floor((real)maxw/2)+1;

    //#pragma omp parallel for private(x1,y1,x2,y2,k)
    for (y1 = 0; y1 < iheight; y1++) {
      for (x1 = 0; x1 < iwidth; x1++) {
	for (y2 = max(0,y1-halfh1); y2 < min(iheight,y1+halfh2); y2++) {
	  for (x2 = max(0,(x1-halfw1)); x2 < min(iwidth,x1+halfw2); x2++) {
	    long dy = y2-y1 + halfh1;
	    long dx = x2-x1 + halfw1;
	    for (k=0; k<ichannels; k++) {
	      real partial_d = 2*(input1_p[k*i1s[0] + y1*i1s[1] + x1*i1s[2]] - input2_p[k*i2s[0] + y2*i2s[1] + x2*i2s[2]]);
	      /*if (partial_d != 0)
		partial_d /= output_p[y1*os[0] + x1*os[1] + dy*os[2] + dx*os[3]];*/
	      partial_d *= gradOutput_p[y1*gos[0] + x1*gos[1] + dy*gos[2] + dx*gos[3]];
	      gradInput1_p[k*gi1s[0] + y1*gi1s[1] + x1*gi1s[2]] += partial_d;
	      gradInput2_p[k*gi2s[0] + y2*gi2s[1] + x2*gi2s[2]] -= partial_d;
	      //real partial_d = 2*(THTensor_(get3d)(input1, k, y1, x1) - THTensor_(get3d)(input2, k, y2, x2));
	      //partial_d *= THTensor_(get4d)(gradOutput, y1, x1, dy, dx);
	      //THTensor_(set3d)(gradInput1, k, y1, x1, THTensor_(get3d)(gradInput1, k, y1, x1) + partial_d);
	      //THTensor_(set3d)(gradInput2, k, y2, x2, THTensor_(get3d)(gradInput2, k, y2, x2) - partial_d);
	    }
	  }
	}
      }
    }
  } else {
    //#pragma omp parallel for private(x1,y1,x2,y2,k)
    for (y1 = 0; y1 < iheight; y1++) {
      for (x1 = 0; x1 < iwidth; x1++) {
	for (y2 = y1; y2 < y1+maxh; y2++) {
	  for (x2 = x1; x2 < x1+maxw; x2++) {
	    for (k = 0; k < ichannels; k++) {
	      real partial_d = 2*(input1_p[k*i1s[0] + y1*i1s[1] + x1*i1s[2]] - input2_p[k*i2s[0] + y2*i2s[1] + x2*i2s[2]]);
	      partial_d *= gradOutput_p[y1*gos[0]+x1*gos[1]+(y2-y1)*gos[2]+(x2-x1)*gos[3]];
	      gradInput1_p[k*gi1s[0] + y1*gi1s[1] + x1*gi1s[2]] += partial_d;
	      gradInput2_p[k*gi2s[0] + y2*gi2s[1] + x2*gi2s[2]] -= partial_d;
	      //real partial_d = 2*(THTensor_(get3d)(input1, k, y1, x1) - THTensor_(get3d)(input2, k, y2, x2));
	      //partial_d *= THTensor_(get4d)(gradOutput, y1, x1, y2-y1, x2-x1);
	      //THTensor_(set3d)(gradInput1, k, y1, x1, THTensor_(get3d)(gradInput1, k, y1, x1) + partial_d);
	      //THTensor_(set3d)(gradInput2, k, y2, x2, THTensor_(get3d)(gradInput2, k, y2, x2) - partial_d);
	      
	    }
	  }
	}
      }
    }
  }

  // done
  return 1;
}

static const struct luaL_Reg nn_(SpatialMatching__) [] = {
  {"SpatialMatching_updateOutput", nn_(SpatialMatching_updateOutput)},
  {"SpatialMatching_updateGradInput", nn_(SpatialMatching_updateGradInput)},
  {NULL, NULL}
};

static void nn_(SpatialMatching_init)(lua_State *L)
{
  luaT_pushmetaclass(L, torch_(Tensor_id));
  luaT_registeratname(L, nn_(SpatialMatching__), "nn");
  lua_pop(L,1);
}

#endif
