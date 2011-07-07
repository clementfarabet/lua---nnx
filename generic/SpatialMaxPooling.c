#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/SpatialMaxPooling.c"
#else

static int nn_(SpatialMaxPooling_forward)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_(Tensor_id));
  int kW = luaT_getfieldcheckint(L, 1, "kW");
  int kH = luaT_getfieldcheckint(L, 1, "kH");
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");
  THTensor *indices = luaT_getfieldcheckudata(L, 1, "indices", torch_(Tensor_id));
  THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_(Tensor_id));
  int threads = luaT_getfieldcheckint(L, 1, "threads");

  luaL_argcheck(L, input->nDimension == 3, 2, "3D tensor expected");
  luaL_argcheck(L, input->size[2] >= kW && input->size[1] >= kH, 2, "input image smaller than kernel size");

  THTensor_(resize3d)(output, input->size[0],
                      (input->size[1] - kH) / dH + 1, 
                      (input->size[2] - kW) / dW + 1 );

  /* indices will contain i,j locatyions for each output point */
  THTensor_(resize4d)(indices, 2,output->size[0],output->size[1],output->size[2]);

  omp_set_num_threads(threads);
  omp_lock_t lock; omp_init_lock(&lock);
  int k,i,j;
  #pragma omp parallel for private(k,i,j)
  for (k = 0; k < input->size[0]; k++)
  {
    THTensor *outputPlane, *inputPlane, *unfoldedInputPlane, *localInput;
    omp_set_lock(&lock);
    inputPlane = THTensor_(new)();
    outputPlane = THTensor_(new)();
    localInput = THTensor_(new)();
    unfoldedInputPlane = THTensor_(new)();

    /* get input and output plane */
    THTensor_(select)(outputPlane, output, 0, k);
    THTensor_(select)(inputPlane, input, 0, k);

    /* Unfold input to get each local window */
    THTensor_(unfold)(unfoldedInputPlane, inputPlane, 0, kH, dH);
    THTensor_(unfold)(unfoldedInputPlane, NULL,       1, kW, dW);
    omp_unset_lock(&lock);

    /* Calculate max points */
    for(i = 0; i < outputPlane->size[0]; i++) {
      for(j = 0; j < outputPlane->size[1]; j++) {
	long maxindex = -1;
	double maxval = -THInf;
	long tcntr = 0;
        int x,y;
        for(y = 0; y < unfoldedInputPlane->size[2]; y++) {
          for(x = 0; x < unfoldedInputPlane->size[3]; x++) {
            double val = THTensor_(get4d)(unfoldedInputPlane, i,j,y,x);
            if (val > maxval) {
              maxval = val;
              maxindex = tcntr;
            }
            tcntr++;
          }
        }

	THTensor_(set4d)(indices,0,k,i,j, (int)(maxindex / dW)+1);
	THTensor_(set4d)(indices,1,k,i,j, (maxindex % dW) +1);
	THTensor_(set2d)(outputPlane,i,j,maxval);
      }
    }

    omp_set_lock(&lock);
    THTensor_(free)(inputPlane);
    THTensor_(free)(outputPlane);
    THTensor_(free)(unfoldedInputPlane);
    THTensor_(free)(localInput);
    omp_unset_lock(&lock);
  }

  /* Cleanup */
  omp_destroy_lock(&lock);

  return 1;
}

static int nn_(SpatialMaxPooling_backward)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_(Tensor_id));
  THTensor *gradOutput = luaT_checkudata(L, 3, torch_(Tensor_id));
  int kW = luaT_getfieldcheckint(L, 1, "kW");
  int kH = luaT_getfieldcheckint(L, 1, "kH");
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");

  THTensor *indices = luaT_getfieldcheckudata(L, 1, "indices", torch_(Tensor_id));
  THTensor *gradInput = luaT_getfieldcheckudata(L, 1, "gradInput", torch_(Tensor_id));

  THTensor *gradOutputPlane, *gradInputPlane, *unfoldedGradInputPlane, *gradLocalInput;
  int k,i,j;

  THTensor_(resizeAs)(gradInput, input);
  THTensor_(zero)(gradInput);

  gradInputPlane = THTensor_(new)();
  gradOutputPlane = THTensor_(new)();
  gradLocalInput = THTensor_(new)();
  unfoldedGradInputPlane = THTensor_(new)();

  for (k = 0; k < input->size[0]; k++)
  {
    /* get input and output plane */
    THTensor_(select)(gradOutputPlane, gradOutput, 0, k);
    THTensor_(select)(gradInputPlane, gradInput, 0, k);

    /* Unfold input to get each local window */
    THTensor_(unfold)(unfoldedGradInputPlane, gradInputPlane, 0, kH, dH);
    THTensor_(unfold)(unfoldedGradInputPlane, NULL,           1, kW, dW);

    /* Calculate max points */
    for(i = 0; i < gradOutputPlane->size[0]; i++) {
      for(j = 0; j < gradOutputPlane->size[1]; j++) {
	THTensor_(select)(gradLocalInput, unfoldedGradInputPlane,0,i);
	THTensor_(select)(gradLocalInput, NULL,                  0,j);
	long maxi = THTensor_(get4d)(indices,0,k,i,j)-1;
	long maxj = THTensor_(get4d)(indices,1,k,i,j)-1;
	double gi = THTensor_(get2d)(gradLocalInput,maxi,maxj)+THTensor_(get2d)(gradOutputPlane,i,j);
	THTensor_(set2d)(gradLocalInput,maxi,maxj,gi);
      }
    }
  }

  /* Cleanup */
  THTensor_(free)(gradInputPlane);
  THTensor_(free)(gradOutputPlane);
  THTensor_(free)(unfoldedGradInputPlane);
  THTensor_(free)(gradLocalInput);

  return 1;
}

static const struct luaL_Reg nn_(SpatialMaxPooling__) [] = {
  {"SpatialMaxPooling_forward", nn_(SpatialMaxPooling_forward)},
  {"SpatialMaxPooling_backward", nn_(SpatialMaxPooling_backward)},
  {NULL, NULL}
};

static void nn_(SpatialMaxPooling_init)(lua_State *L)
{
  luaT_pushmetaclass(L, torch_(Tensor_id));
  luaT_registeratname(L, nn_(SpatialMaxPooling__), "nn");
  lua_pop(L,1);
}

#endif
