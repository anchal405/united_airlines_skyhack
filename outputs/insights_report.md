# Operational Insights

## Destinations that trend difficult

- NAS: avg difficulty 0.62, 100.0% flights classified as Difficult (n=4).
- GTF: avg difficulty 0.61, 100.0% flights classified as Difficult (n=2).
- YHZ: avg difficulty 0.61, 100.0% flights classified as Difficult (n=2).
- AUA: avg difficulty 0.61, 75.0% flights classified as Difficult (n=4).
- PLS: avg difficulty 0.60, 100.0% flights classified as Difficult (n=2).

## Common drivers (overall model)

- transfer_bag_ratio: coefficient 0.050 (standardized)
- special_needs_ratio: coefficient 0.050 (standardized)
- pax_load_factor: coefficient 0.037 (standardized)
- neg_slack_minutes: coefficient 0.036 (standardized)
- basic_econ_ratio: coefficient 0.000 (standardized)
- Special-needs detection has been enhanced using SSR (wheelchair, UMNR, medical) data from PNR Remarks, improving accuracy of operational difficulty estimation.

## Drivers for the hardest destinations

- AUA: top drivers → transfer_bag_ratio (0.061); special_needs_ratio (0.017); neg_slack_minutes (0.015) [n=4]
- NAS: top drivers → special_needs_ratio (0.028); transfer_bag_ratio (0.024); neg_slack_minutes (0.006) [n=4]

## Geographical Insights

- BS: avg difficulty 0.62, 100.0% flights Difficult.
- AW: avg difficulty 0.61, 75.0% flights Difficult.
- TC: avg difficulty 0.60, 100.0% flights Difficult.
- KY: avg difficulty 0.56, 50.0% flights Difficult.
- JM: avg difficulty 0.52, 57.1% flights Difficult.

## Operational recommendations

- **Tight turns / negative slack** → Add buffer minutes to schedule on specific turns; pre-stage turn teams; coordinate fueling/catering earlier.
- **High transfer bag ratio** → Pre-position bag transfer staff; tighten minimum connect times; priority belt allocation for connections.
- **High passenger load factor** → Add an extra gate agent at boarding; enforce boarding groups; open a second document check line.
- **High special needs ratio** → Pre-assign wheelchair handlers and aisle chairs; brief crew; allow earlier boarding for SSR passengers.
- **High basic-economy mix** → Prepare for boarding exceptions (seating/baggage); proactive signage; pre-brief gate agents on exception handling.