# WM-811K wafer feature contract (schema version 1)

One row represents one labeled source wafer, identified by the original unique
pickle index (`wafer_id`). `failureType` is the supervised target: Center, Donut,
Edge-Loc, Edge-Ring, Loc, Near-full, Random, Scratch, or `none`. `none` is a labeled
normal wafer; it is not an unlabeled wafer. Missing, empty and unknown labels are
excluded. `lotName` and the dataset's misspelled `trianTestLabel` are retained as
metadata and never passed to the classifier. Die size and wafer index are not
model inputs.

Maps must be nonempty rectangular 2D arrays containing only 0 (empty), 1 (pass),
and 2 (fail). Invalid maps raise an error rather than silently dropping wafers.
All counts exclude zeros. Basic features are map height/width, tested die count,
passing/failing counts, yield = passing/tested, and failure rate = failing/tested.
For maps with no tested dies both rates are defined as zero; exclude these maps
when interpreting yield. The dashboard does so for the yield histogram.

Coordinates are centered on the rectangular map midpoint. Each axis is divided
by max((axis length - 1)/2, 1), giving approximately [-1, 1], with x increasing
rightward and y downward. Radius r = sqrt(x²+y²); corners may exceed radius 1.
These coordinates describe the map bounding rectangle, not a fitted physical
wafer circle. Odd-dimension center rows/columns belong to south/east quadrants.

| Feature | Definition |
|---|---|
| center_failure_density | Failed / tested dies with r <= 0.5 |
| edge_failure_density | Failed / tested dies with r >= 0.8 |
| nw/ne/sw/se_failure_rate | Failed / tested within the indicated quadrant |
| failure_centroid_x/y | Mean normalized coordinates of failing dies |
| failure_radius_mean/std | Mean and population standard deviation of failing-die radius |
| failure_spread | Sum of population x/y coordinate variances among failing dies |
| inner/middle/outer_failure_fraction | Fraction of failing dies at r <= 0.5, 0.5 < r < 0.8, r >= 0.8 |
| adjacent_failure_fraction | Fraction of failing dies with at least one failing orthogonal neighbor (no wrapping) |

Empty regional denominators produce zero. All failure-location features are zero
when there are no failures. Failure count/rate distinguish this convention from
an actual centered failure. Adjacency is a local concentration indicator, not a
connected-component count or a calibrated spatial significance test. Resolution
and die layout affect adjacency even after coordinate normalization.

The feature schema has 22 numerical features, plus wafer_id, failureType,
lotName and trianTestLabel. The feature list used by training is explicitly
constructed from the extractor, preventing accidental use of identifiers or
labels. Correlated yield/rate/count features are intentional in this baseline;
forest impurity importance can split credit between correlated features and
favor continuous variables. Importance does not establish physical causality.
