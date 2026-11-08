# Semiconductor Test Data Contract

## Record Definition

One row represents one device measurement under one
specific operating condition during a test run.

## Identification Fields

- lot_id
- wafer_id
- device_id
- test_id
- test_condition_id

## Operating Conditions

- voltage_v
- frequency_mhz
- temperature_c

## Electrical Measurements

- current_ma
- power_mw
- leakage_current_ua
- propagation_delay_ns
- performance_score

## Classification Fields

- test_status
- failure_category

## Data Integrity Requirements

1. Device identifiers must be consistent.
2. Each record must reference a valid lot and wafer.
3. Operating conditions must be within configured ranges.
4. Numerical measurements must be valid.
5. Power measurements must be consistent with the
   voltage and current measurement definitions.
6. Failure labels must follow documented simulation rules.
7. Missing and anomalous values must be handled explicitly.
8. Dataset generation must be reproducible using a fixed seed.

## Initial Dataset Target

1,000,000 measurement records.

Storage formats:

- CSV for interoperability.
- Parquet for analytical processing.

All generated records must be identified as synthetic.
