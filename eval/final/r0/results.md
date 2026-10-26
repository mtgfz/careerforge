# CareerForge evaluation results

5 CV/offer pairs, 15 successful runs.

## Summary (mean ± std per variant)

| variant     | faithfulness   | ats           | quality       | overall       |   pass_rate |   n_unsupported |   pruned |   iterations |   cv_coverage |   seconds |   runs |
|:------------|:---------------|:--------------|:--------------|:--------------|------------:|----------------:|---------:|-------------:|--------------:|----------:|-------:|
| single_pass | 0.950 ± 0.046  | 0.422 ± 0.165 | 0.780 ± 0.045 | 0.741 ± 0.060 |         0.4 |             1.2 |      0   |            1 |         0.655 |     27.06 |      5 |
| critic_once | 1.000 ± 0.000  | 0.425 ± 0.189 | 0.760 ± 0.055 | 0.755 ± 0.052 |         0.4 |             0   |      0.6 |            1 |         0.632 |     75.4  |      5 |
| full        | 1.000 ± 0.000  | 0.402 ± 0.158 | 0.780 ± 0.045 | 0.755 ± 0.056 |         0.4 |             0   |      0.2 |            2 |         0.665 |     75.22 |      5 |

## Faithfulness: paired comparison vs `single_pass`

| variant     |   pairs |   mean_delta |   wins |   ties |   losses |   sign_test_p |
|:------------|--------:|-------------:|-------:|-------:|---------:|--------------:|
| critic_once |       5 |         0.05 |      4 |      1 |        0 |         0.125 |
| full        |       5 |         0.05 |      4 |      1 |        0 |         0.125 |

_Wins/losses count pairs where the variant beats/trails the baseline by more than 0.005; p is a two-sided exact sign test (few pairs: read it as a trend, not a proof)._

## Overall: paired comparison vs `single_pass`

| variant     |   pairs |   mean_delta |   wins |   ties |   losses |   sign_test_p |
|:------------|--------:|-------------:|-------:|-------:|---------:|--------------:|
| critic_once |       5 |        0.014 |      3 |      1 |        1 |         0.625 |
| full        |       5 |        0.014 |      4 |      0 |        1 |         0.375 |

_Wins/losses count pairs where the variant beats/trails the baseline by more than 0.005; p is a two-sided exact sign test (few pairs: read it as a trend, not a proof)._

## Faithfulness per pair

| pair                  |   single_pass |   critic_once |   full |
|:----------------------|--------------:|--------------:|-------:|
| 01_ia_agents_emsi     |         0.967 |             1 |      1 |
| 02_fullstack_cnexia   |         0.963 |             1 |      1 |
| 03_data_analyst_alten |         0.875 |             1 |      1 |
| 04_support_pca        |         1     |             1 |      1 |
| 05_poor_fit_artefact  |         0.947 |             1 |      1 |

## Overall per pair

| pair                  |   single_pass |   critic_once |   full |
|:----------------------|--------------:|--------------:|-------:|
| 01_ia_agents_emsi     |         0.774 |         0.773 |  0.782 |
| 02_fullstack_cnexia   |         0.826 |         0.833 |  0.834 |
| 03_data_analyst_alten |         0.674 |         0.743 |  0.73  |
| 04_support_pca        |         0.722 |         0.733 |  0.688 |
| 05_poor_fit_artefact  |         0.709 |         0.695 |  0.739 |

## Critic loop trajectory (`full`)

|   iteration |   faithfulness |   ats |   quality |   overall |   runs |
|------------:|---------------:|------:|----------:|----------:|-------:|
|           1 |          0.938 | 0.425 |      0.78 |     0.737 |      5 |
|           2 |          0.977 | 0.358 |      0.75 |     0.723 |      4 |
|           3 |          1     | 0.33  |      0.8  |     0.739 |      1 |
