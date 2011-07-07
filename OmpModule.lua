local OmpModule,parent = torch.class('nn.OmpModule','nn.Module')

function OmpModule:__init()
   parent.__init(self)
   self.threads = 1
end

function OmpModule:read(file)
   parent.read(self, file)
   self.threads = 1
end
