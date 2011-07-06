local SpatialGraph, parent = torch.class('nn.SpatialGraph', 'nn.Module')

local help_desc =
[[Creates an edge-weighted graph from a set of N feature
maps. 

The input is a 3D tensor width x height x nInputPlane, the
output is a 3D tensor width x height x 2. The first slice
of the output contains horizontal edges, the second vertical
edges.

The input features are assumed to be >= 0.
More precisely:
+ dist == 'euclid' and norm == true: the input features should 
  also be <= 1, to produce properly normalized distances (btwn 0 and 1);
+ dist == 'cosine': the input features do not need to be bounded, 
  as the cosine dissimilarity normalizes with respect to each vector.
  An epsilon is automatically added, so that components that are == 0
  are properly considered as being similar.
]]

function SpatialGraph:__init(...)
   parent.__init(self)

   xlua.unpack_class(self, {...}, 'nn.SpatialGraph',  help_desc,
                        {arg='dist', type='string', help='distance metric to use', default='euclid'},
                        {arg='normalize', type='boolean', help='normalize euclidean distances btwn 0 and 1 (assumes input range to be btwn 0 and 1)', default=true},
                        {arg='connex', type='number', help='connexity', default=4})
   
   _ = (self.connex == 4) or xerror('4 is the only connexity supported, for now',
                                    'nn.SpatialGraph',self.usage)
   self.dist = ((self.dist == 'euclid') and 0) or ((self.dist == 'cosine') and 1)
       or xerror('euclid is the only distance supported, for now','nn.SpatialGraph',self.usage)
   self.normalize = (self.normalize and 1) or 0

   if self.dist == 'cosine' and self.normalize == 1 then
      xerror('normalized cosine is not supported for now [just because I couldnt figure out the gradient :-)]',
             'nn.SpatialGraph', self.usage)
   end
end

-- C macros
inline.preamble [[
#define square(x) ((x)*(x))
]]

-- define fprop in C
SpatialGraph.forward_c = inline.load [[
      // get all params
      const void* torch_Tensor_id = luaT_checktypename2id(L, "torch.Tensor");
      THTensor *input = luaT_checkudata(L, 2, torch_Tensor_id);
      int connex = luaT_getfieldcheckint(L, 1, "connex");
      int dist = luaT_getfieldcheckint(L, 1, "dist");
      int norm = luaT_getfieldcheckint(L, 1, "normalize");
      THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_Tensor_id);

      // dims
      int iwidth = input->size[0];
      int iheight = input->size[1];
      int ichannels = input->size[2];
      int owidth = iwidth;
      int oheight = iheight;
      int ochannels = connex / 2;

      // norm ?
      double normer = (norm == 1) ? 1/sqrt(ichannels) : 1;

      // zero output
      THTensor_zero(output);

      // Euclidean distance
      if (dist == 0) {
         // Sum[ (Xi - Xi+1)^2 ]
         int x,y,k;
         for (k=0; k<ichannels; k++) {
            for (y=0; y<oheight; y++) {
               for (x=0; x<owidth; x++) {
                  if (x < owidth-1) {
                     double temp = square(THTensor_get3d(input, x, y, k) - THTensor_get3d(input, x+1, y, k));
                     THTensor_set3d(output, x, y, 0, temp + THTensor_get3d(output, x, y, 0));
                  }
                  if (y < oheight-1) {
                     double temp = square(THTensor_get3d(input, x, y, k) - THTensor_get3d(input, x, y+1, k));
                     THTensor_set3d(output, x, y, 1, temp + THTensor_get3d(output, x, y, 1));
                  }
               }
            }
         }

         // Sqrt[ Sum[ (Xi - Xi+1)^2 ] ]
         for (k=0; k<ochannels; k++) {
            for (y=0; y<oheight; y++) {
               for (x=0; x<owidth; x++) {
                  THTensor_set3d(output, x, y, k, sqrt(THTensor_get3d(output, x, y, k)) * normer);
               }
            }
         }

      // Cosine dissimilarity
      } else {
         // add epsilon to input (to get rid of 0s)
         THTensor *inputb = THTensor_newWithSize3d(input->size[0], input->size[1], input->size[2]);
         THTensor_copy(inputb, input);
         THTensor_add(inputb, 1e-12);

         // Sum[ (Xi * Xi+1) ]
         int x,y,k;
         for (y=0; y<oheight; y++) {
            for (x=0; x<owidth; x++) {
               double norm_A = 0;
               double norm_B = 0;
               double norm_C = 0;
               for (k=0; k<ichannels; k++) {
                  norm_A += square(THTensor_get3d(inputb, x, y, k));
                  if (x < owidth-1) {
                     double temp = THTensor_get3d(inputb, x, y, k) * THTensor_get3d(inputb, x+1, y, k);
                     THTensor_set3d(output, x, y, 0, temp + THTensor_get3d(output, x, y, 0));
                     norm_B += square(THTensor_get3d(inputb, x+1, y, k));
                  }
                  if (y < oheight-1) {
                     double temp = THTensor_get3d(inputb, x, y, k) * THTensor_get3d(inputb, x, y+1, k);
                     THTensor_set3d(output, x, y, 1, temp + THTensor_get3d(output, x, y, 1));
                     norm_C += square(THTensor_get3d(inputb, x, y+1, k));
                  }
               }
               if (x < owidth-1) {
                  if (norm) {
                     THTensor_set3d(output, x, y, 0, 1 - THTensor_get3d(output, x, y, 0) / (sqrt(norm_A) * sqrt(norm_B)));
                  } else {
                     THTensor_set3d(output, x, y, 0, ichannels - THTensor_get3d(output, x, y, 0));
                  }
               }
               if (y < oheight-1) {
                  if (norm) {
                     THTensor_set3d(output, x, y, 1, 1 - THTensor_get3d(output, x, y, 1) / (sqrt(norm_A) * sqrt(norm_C)));
                  } else {
                     THTensor_set3d(output, x, y, 1, ichannels - THTensor_get3d(output, x, y, 1));                   
                  }
               }
            }
         }

         // Cleanup
         THTensor_free(inputb);
      }

      return 1;
]]

-- define bprop in C
SpatialGraph.backward_c = inline.load [[
      // get all params
      const void* torch_Tensor_id = luaT_checktypename2id(L, "torch.Tensor");
      THTensor *input = luaT_checkudata(L, 2, torch_Tensor_id);
      THTensor *gradInput = luaT_getfieldcheckudata(L, 1, "gradInput", torch_Tensor_id);
      THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_Tensor_id);
      THTensor *gradOutput = luaT_checkudata(L, 3, torch_Tensor_id);
      int connex = luaT_getfieldcheckint(L, 1, "connex");
      int dist = luaT_getfieldcheckint(L, 1, "dist");
      int norm = luaT_getfieldcheckint(L, 1, "normalize");

      // dims
      int iwidth = input->size[0];
      int iheight = input->size[1];
      int ichannels = input->size[2];
      int owidth = gradOutput->size[0];
      int oheight = gradOutput->size[1];
      int ochannels = gradOutput->size[2];

      // norm ?
      double normer = (norm == 1) ? 1/sqrt(ichannels)/sqrt(ichannels) : 1;

      // resize gradInput
      THTensor_zero(gradInput);

      // compute derivatives, and backpropagate output error to input
      if (dist == 0) {
         int x,y,k;
         for (k=0; k<ichannels; k++) {
            for (y=0; y<oheight; y++) {
               for (x=0; x<owidth; x++) {
                  if (x < owidth-1) {
                     double partial_d = THTensor_get3d(input, x, y, k) - THTensor_get3d(input, x+1, y, k);
                     if (partial_d != 0) partial_d /= THTensor_get3d(output, x, y, 0);
                     partial_d *= THTensor_get3d(gradOutput, x, y, 0) * normer;
                     THTensor_set3d(gradInput, x, y, k, partial_d + THTensor_get3d(gradInput, x, y, k));
                     THTensor_set3d(gradInput, x+1, y, k, -partial_d + THTensor_get3d(gradInput, x+1, y, k));
                  }
                  if (y < oheight-1) {
                     double partial_d = THTensor_get3d(input, x, y, k) - THTensor_get3d(input, x, y+1, k);
                     if (partial_d != 0) partial_d /= THTensor_get3d(output, x, y, 1);
                     partial_d *= THTensor_get3d(gradOutput, x, y, 1) * normer;
                     THTensor_set3d(gradInput, x, y, k, partial_d + THTensor_get3d(gradInput, x, y, k));
                     THTensor_set3d(gradInput, x, y+1, k, -partial_d + THTensor_get3d(gradInput, x, y+1, k));
                  }
               }
            }
         }

      // Cosine
      } else {
         int x,y,k;
         for (y=0; y<oheight; y++) {
            for (x=0; x<owidth; x++) {
               double sum_A = 0;
               double sum_B = 0;
               double sum_C = 0;
               double sum_AB = 0;
               double sum_AC = 0;

               if (norm) {
                  for (k=0; k<ichannels; k++) {
                     sum_A += square(THTensor_get3d(input, x, y, k));
                     if (x < owidth-1) {
                        sum_B += square(THTensor_get3d(input, x+1, y, k));
                        sum_AB += THTensor_get3d(input, x, y, k) * THTensor_get3d(input, x+1, y, k);
                     }
                     if (y < oheight-1) {
                        sum_C += square(THTensor_get3d(input, x, y+1, k));
                        sum_AC += THTensor_get3d(input, x, y, k) * THTensor_get3d(input, x, y+1, k);
                     }
                  }
               }

               double term1, term2, term3, partial_d;
               double epsi = 1e-12;
               if (x < owidth-1) {
                  if (norm) {
                     term1 = 1 / ( pow(sum_A, 1/2) * pow(sum_B, 1/2) + epsi );
                     term2 = sum_AB / ( pow(sum_A, 3/2) * pow(sum_B, 1/2) + epsi );
                     term3 = sum_AB / ( pow(sum_B, 3/2) * pow(sum_A, 1/2) + epsi );
                  }
                  for (k=0; k<ichannels; k++) {
                     if (norm) {
                        partial_d = term2 * THTensor_get3d(input, x, y, k) 
                                  - term1 * THTensor_get3d(input, x+1, y, k);
                     } else {
                        partial_d = -THTensor_get3d(input, x+1, y, k);
                     }
                     partial_d *= THTensor_get3d(gradOutput, x, y, 0);
                     THTensor_set3d(gradInput, x, y, k, partial_d + THTensor_get3d(gradInput, x, y, k));

                     if (norm) {
                        partial_d = term3 * THTensor_get3d(input, x+1, y, k) 
                                  - term1 * THTensor_get3d(input, x, y, k);
                     } else {
                        partial_d = -THTensor_get3d(input, x, y, k);
                     }
                     partial_d *= THTensor_get3d(gradOutput, x, y, 0);
                     THTensor_set3d(gradInput, x+1, y, k, partial_d + THTensor_get3d(gradInput, x+1, y, k));
                  }
               }
               if (y < oheight-1) {
                  if (norm) {
                     term1 = 1 / ( pow(sum_A, 1/2) * pow(sum_C, 1/2) + epsi );
                     term2 = sum_AC / ( pow(sum_A, 3/2) * pow(sum_C, 1/2) + epsi );
                     term3 = sum_AC / ( pow(sum_C, 3/2) * pow(sum_A, 1/2) + epsi );
                  }
                  for (k=0; k<ichannels; k++) {
                     if (norm) {
                        partial_d = term2 * THTensor_get3d(input, x, y, k)
                                  - term1 * THTensor_get3d(input, x, y+1, k);
                     } else {
                        partial_d = -THTensor_get3d(input, x, y+1, k);
                     }
                     partial_d *= THTensor_get3d(gradOutput, x, y, 1);
                     THTensor_set3d(gradInput, x, y, k, partial_d + THTensor_get3d(gradInput, x, y, k));

                     if (norm) {
                        partial_d = term3 * THTensor_get3d(input, x, y+1, k) 
                                  - term1 * THTensor_get3d(input, x, y, k);
                     } else {
                        partial_d = -THTensor_get3d(input, x, y, k);
                     }
                     partial_d *= THTensor_get3d(gradOutput, x, y, 1);
                     THTensor_set3d(gradInput, x, y+1, k, partial_d + THTensor_get3d(gradInput, x, y+1, k));
                  }
               }
            }
         }
      }

      return 1;
]]

function SpatialGraph:forward(input)
   self.output:resize(input:size(1), input:size(2), self.connex / 2)
   self:forward_c(input)
   return self.output
end

function SpatialGraph:backward(input, gradOutput)
   self.gradInput:resizeAs(input)
   self:backward_c(input, gradOutput)
   return self.gradInput
end

function SpatialGraph:write(file)
   parent.write(self, file)
   file:writeInt(self.connex)
   file:writeInt(self.dist)
   file:writeInt(self.normalize)
end

function SpatialGraph:read(file)
   parent.read(self, file)
   self.connex = file:readInt()
   self.dist = file:readInt()
   self.normalize = file:readInt()
end
