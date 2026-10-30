# GMKtec rear-panel reference

`source.png` is the unmodified user-supplied photograph. `landmarks.json` records manual
case-envelope corners and connector centers. `scenes/pc_rear.py` maps the original image
onto a tessellated visual mesh using a projective UV transform; no generated image or
bitmap retouching is used. The measured case envelope is 102.5 x 41.2 mm. Depth is 97 mm.
Feet raise total height to 44.5 mm. Rounded case edges and photographic perspective make
this a visual registration, not exact port-center metrology.

The second USB-C center is the physical fixture origin. Photo-derived offsets set the case,
feet and inertia center relative to this origin consistently. The original receptacle and
plug dimensions are unchanged. The textured face is non-colliding and omits the target
aperture, leaving its physical cavity accessible. Other ports remain visual-only.

The new reader caliper body length (29.6 mm) is the SILVER inset, confirmed by the user.
The black outer housing must also exist. Current measured scene preserves the earlier
90.46 mm total and 34.73 x 8.83 mm outer body envelope; it uses the new 43.4 x 12.96 x 7 mm
neck and 17 g mass. The derived outer housing axial length is 41.26 mm.
The unchanged 6.65 mm metal plug is embedded 0.85 mm into the overmold, leaving
the measured 5.8 mm exposed. The neck and housing placements, grasp centre, and
neck seam follow this embedding. The measured bimanual seating criterion is
limited by the overmold at 5.8 mm; the cavity remains 6.5 mm deep. This is a
geometric stop, not validation of insertion forces or electrical connection.
Silver inset is 34.6 x 29.6 mm; its axial placement remains estimated.
The 41.26 mm outer housing length is derived from the total, not independently measured.


Legacy fixed-scene defaults are kept separate; these measurements currently drive the
new free-PC scene, not every older training configuration.
