# TEAM TODO - delete this file before building the submission zip

The code runs end to end. What is left needs HUMANS (and real data):

1. [ ] Put the Kaggle CSV in `Dataset/Raw/`, then `python Main.py all`. Read every printed table.
2. [ ] Check the scheme-vs-label table from `explore`. If lopsided, keep `USE_SCHEME_FEATURES = False` and say so in the report.
3. [ ] Additional data (4 marks): download Tranco top-sites and/or an OpenPhish/PhishTank feed into `Dataset/Raw/Extra/` (file-name rules in README), re-run `process`. Describe the shape mismatch (bare domains vs full URLs) and `URL_Parser.py` as the transformation.
4. [ ] Models beyond the unit (4 marks): edit `BEYOND_UNIT` in `src/Models.py` to match what COS30049 taught; make sure xgboost + lightgbm are installed; write WHY each suits URL data.
5. [ ] Clusters: open `reports/cluster_profiles.md`, write the interpretation under each cluster using the z-score table and examples (brief: describe by what members have in common / what is missing).
6. [ ] Error analysis: open `reports/error_analysis.csv`, read ~10 false positives and ~10 false negatives by hand, write what they have in common.
7. [ ] Check Canvas for the AI-use rule in the brief before using any AI-written text in the report.
8. [ ] Report <= 4000 words (see docs/feature_justification.md for feature reasoning + candidate references - verify them).
9. [ ] A2 minutes + A2 contribution form (all three sign together), file names per the brief.
10. [ ] Zip: code, README, `Dataset/Processed/*.csv` (only data used by the final model), `models/*.joblib`. No raw data, no TEAM_TODO.md, no .DS_Store.
