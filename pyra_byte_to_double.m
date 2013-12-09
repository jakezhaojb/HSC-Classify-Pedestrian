
function pyra=pyra_byte_to_double(pyra)

if isempty(pyra), return; end;
if isfield(pyra,'feat'),
  use_feat=1;
  feat=pyra.feat;
else
  use_feat=0;
  feat=pyra;
end

nscale=length(feat);
for i=1:nscale,
  feat{i}=double(feat{i})/255;
end

if use_feat,
  pyra.feat=feat;
else
  pyra=feat;
end


