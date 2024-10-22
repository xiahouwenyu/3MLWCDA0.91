from ..HAL import HAL

from threeML import DataList, JointLikelihood


def fit_point_source(roi,
                     maptree,
                     response,
                     point_source_model,
                     bin_list,
                     confidence_intervals=False,
                     liff=False,
                     pixel_size=0.17,
                     verbose=False):

    data_radius = roi.data_radius.to("deg").value

    if not liff:

        # This is a 3ML plugin
        WCDA = HAL("WCDA",
                   maptree,
                   response,
                   roi,
                   flat_sky_pixels_size=pixel_size)

        WCDA.set_active_measurements(bin_list=bin_list)

    else:

        from threeML import WCDALike

        WCDA = WCDALike("WCDA",
                        maptree,
                        response,
                        fullsky=True)

        WCDA.set_bin_list(bin_list)

        ra_roi, dec_roi = roi.ra_dec_center

        WCDA.set_ROI(ra_roi, dec_roi, data_radius)

    if not liff:

        WCDA.display()

    data = DataList(WCDA)

    jl = JointLikelihood(point_source_model, data, verbose=verbose)

    point_source_model.display(complete=True)

    try:

        jl.set_minimizer("minuit")

    except:

        jl.set_minimizer("minuit")

    param_df, like_df = jl.fit()

    if confidence_intervals:

        ci = jl.get_errors()

    else:

        ci = None

    return param_df, like_df, ci, jl.results