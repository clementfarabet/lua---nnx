#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/lbfgs.c"
#else

/* use storages to allow copying from different types of Tensors on
   the function evaluations side.  The lbfgs operations are hard
   coded to use doubles for now. lbfgsfloatval_t = double */

int THTensor_(copy_evaluate_start)(THTensor *parameters,
                                   const lbfgsfloatval_t *x, 
                                   const int nParameter)
{
  THDoubleStorage *xs = THDoubleStorage_newWithData((double *)x,nParameter);
  THStorage *ps = 
    THStorage_(newWithData)(THTensor_(data)(parameters),nParameter);

 
  /* copy given x (xs) -> parameters (ps) */
  THStorage_(copyDouble)(ps,xs);

  /* only want to free the struct part of the storage not the data */
  xs->data = NULL;
  THDoubleStorage_free(xs);
  ps->data = NULL;
  THStorage_(free)(ps);
  return 0;
}

int THTensor_(copy_evaluate_end)(lbfgsfloatval_t *g, 
                                 const THTensor * gradParameter,
                                 const int nParameter)
{
  THDoubleStorage *gs = THDoubleStorage_newWithData((double *)g,nParameter);
  THStorage *gps = 
    THStorage_(newWithData)(THTensor_(data)(gradParameters), nParameter);
  
  /* copy gradParameters (gps) -> g (gs) */
#ifdef TH_REAL_IS_FLOAT
  THDoubleStorage_copyFloat(gs,gps);
#else
#ifdef TH_REAL_IS_CUDA
  THDoubleStorage_copyCuda(gs,gps);
#else
  THDoubleStorage_copy(gs,gps);
#endif
#endif
  /* only want to free the struct part of the storage not the data */
  gs->data = NULL;
  THDoubleStorage_free(gs);
  gps->data = NULL;
  THStorage_(free)(gps);

  return 0;
}


int THTensor_(copy_init)(lbfgsfloatval_t *x, 
                         THTensor *parameters,
                         const int nParameter) 
{
  THDoubleStorage *xs = THDoubleStorage_newWithData((double *)x,nParameter);
  THStorage *ps = 
    THStorage_(newWithData)(THTensor_(data)(parameters),nParameter);

  /* copy given parameters (ps) -> x (xs) */
#ifdef TH_REAL_IS_FLOAT
  THDoubleStorage_copyFloat(xs,ps);
#else
#ifdef TH_REAL_IS_CUDA
  THDoubleStorage_copyCuda(xs,ps);
#else 
  THDoubleStorage_copy(xs,ps);
#endif
#endif 
  /* only want to free the struct part of the storage not the data */
  xs->data = NULL;
  THDoubleStorage_free(xs);
  ps->data = NULL;
  THStorage_(free)(ps);
  printf("in copy_init : freed storage\n");

  /* done */
  return 0;
}

#endif
