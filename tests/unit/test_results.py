"""Fast tests for the BOResults container. No GP training."""
import pandas as pd

from emcal import BOResults


def test_boresults_accepts_none_and_stores_fields():
    # BOResults is a plain record; every field is optional (None) so partial results
    # (e.g. the split simple/GP result objects the driver builds) construct cleanly.
    df = pd.DataFrame({"bo_iter": [1, 2], "best_sse_actual": [1.0, 0.5]})
    res = BOResults(
        configuration={"cs_name_val": 1},
        simulator_class=None,
        exp_data_class=None,
        list_gp_emulator_class=None,
        results_df=df,
        max_ei_details_df=None,
        why_term="converged",
        heat_map_data_dict=None,
    )
    assert res.configuration == {"cs_name_val": 1}
    assert res.why_term == "converged"
    assert res.results_df.shape == (2, 2)
    assert res.list_gp_emulator_class is None
