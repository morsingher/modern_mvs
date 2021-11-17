# modern_mvs

CURRENT SITUATION (16/11/2021):

- ACMH with geometric consistency works and it's better than previous version. 
- Multi-scale reasoning from ACMM seems to work, but apparently it doesn't improve much. However, it is still experimental and JBU is not implemented. Furthermore, it might be possible that I just tested on a dataset (the statue) where this difference is not relevant. Try on pipes maybe.
- Planar priors from ACMP are implemented but there is some bug. I don't understand if it's a bug in my code on in their math, because there were a lot of bugs in their code also. In the short term, we should fix this. In the long term, I think this part should be just removed and replaced with the global refinement step.

To sum up -> disable planar priors in config.json and set max_size = min_size to disable multi-scale stuff.

TODO LIST:

- You should save things with postfix photo/planar/geom
- There are a set of TODOs in the code, which represent either bad or unclear parts. Look for them and fix them.
- Rename io_utils.* to io.*
- Make all functions CamelCase
- Include headers with correct folder
- General cleanup of unnecessary headers
- Make use of unit tests where possible
- Set `__restrict__` where applicable
- Is it good practice to throw std::runtime_error() instead of checking return values?
- Maybe move IO stuff into relevant classes (i.e. Mat and Options)?
- A lot of stuff in CUDA kernels should be moved to config.json
- Class design is probably poor and should be improved both in terms of constructor, destructor etc, and for public/private stuff
- Camera could be header only and (w,h) should be passed to the constructor
- Start Doxygen docs
- I think that passing costs to CUDA for geometric consistency is useless since cost gets recomputed in any case
- Add refinement to scripts/python, add code to call it and modify it so that it accepts and generates the correct format

Then, in the long term:

- Include support for all your new stuff (d-n consistency, keypoint initialization, refinement and so on)