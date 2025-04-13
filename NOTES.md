# Notes written in blood
- When using `slices` syntax to specify a range of layers, `layer_range` is `(a,b]` (ie, returns layers a+1 to b), ***not*** a fully inclusive `[a,b]` (returns layers a to b) or anything else. The layers are also one-indexed, as far as i can tell. for some reason. fuck you.
