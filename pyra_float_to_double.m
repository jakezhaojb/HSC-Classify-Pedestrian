
function pyra=float_to_double(pyra)

nscale=length(pyra.feat);
for i=1:nscale,
  pyra.feat{i}=double(pyra.feat{i});
end
