-- Template SQL for first-24h ICU summary extraction (MIMIC-IV schema)
-- STRICT 0–24h window; DO NOT include any data after 24h
-- WARNING: Avoid label leakage (e.g., post-24h labs, interventions)
-- TODO: Replace synthetic with local MIMIC path (never committed)

SELECT
    subject_id,
    hadm_id,
    icustay_id,
    EXTRACT(EPOCH FROM (icu_intime - admit_time))/3600 AS admit_to_icu_hours,
    -- ...extract features for first 24h only...
    -- TODO: add feature columns
FROM
    mimiciv_icu.icustays
WHERE
    icu_intime >= admit_time
    AND icu_outtime <= admit_time + INTERVAL '24 hours';
