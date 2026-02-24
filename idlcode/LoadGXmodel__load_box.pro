function LoadGXmodel__load_box, infile
 inlower=strlowcase(string(infile))
 ext=strsplit(inlower, '.', /extract)
 ext=ext[-1]
 if (ext eq 'sav') or (ext eq 'xdr') then begin
  restore, infile
  return, box
 endif
 if (ext eq 'h5') or (ext eq 'hdf5') then begin
  resolve_routine, 'ConvertToGX', /either
  return, ConvertToGX(infile)
 endif
 message, 'Unsupported model format: '+infile
end