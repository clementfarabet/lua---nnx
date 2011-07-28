local OmpModule,parent = torch.class('nn.OmpModule','nn.Module')

function OmpModule:__init()
   parent.__init(self)
   if openmp then
      self.threads = openmp.getNumThreads()
   else
      self.threads = 1
   end
end

function OmpModule:read(file)
   parent.read(self, file)
   if openmp then
      self.threads = openmp.getNumThreads()
   else
      self.threads = 1
   end
end
