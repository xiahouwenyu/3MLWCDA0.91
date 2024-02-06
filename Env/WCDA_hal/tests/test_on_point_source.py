from WCDA_hal import HealpixConeROI
from conftest import fit_point_source
import argparse


def test_on_point_source(roi,
                         maptree,
                         response,
                         point_source_model,
                         liff=False):

    fit_point_source(roi,
                      maptree,
                      response,
                      point_source_model,
                      liff=liff)


if __name__ == "__main__":
    from conftest import point_source_model, maptree, response
    from conftest import roi as default_roi

    def_roi = default_roi()

    def_model = point_source_model()

    def_ra, def_dec = def_model.pts.position.ra.value, def_model.pts.position.dec.value
    def_mrad = def_roi.model_radius
    def_drad = def_roi.data_radius
    def_ra_roi, def_dec_roi = def_roi.ra_dec_center

    parser = argparse.ArgumentParser()
    parser.add_argument("--liff", help="Use LIFF instead of HAL (for benchmarking)", action="store_true")
    parser.add_argument("--ra", help="RA of source", type=float, default=def_ra)
    parser.add_argument("--dec", help="Dec of source", type=float, default=def_dec)
    parser.add_argument("--ra_roi", help="RA of center of ROI", type=float, default=def_ra_roi)
    parser.add_argument("--dec_roi", help="Dec of center of ROI", type=float, default=def_dec_roi)
    parser.add_argument("--data_radius", help="Radius of the data ROI", type=float, default=def_drad)
    parser.add_argument("--model_radius", help="Radius of the model ROI", type=float, default=def_mrad)
    parser.add_argument("--maptree", help="Maptree", type=str, default=maptree())
    parser.add_argument("--response", help="Response", type=str, default=response())
    parser.add_argument("--free_position", help='Use this to set the position free', action='store_true', default=False)

    args = parser.parse_args()

    roi = HealpixConeROI(data_radius=args.data_radius,
                         model_radius=args.model_radius,
                         ra=args.ra_roi,
                         dec=args.dec_roi)

    pts_model = point_source_model(ra=args.ra, dec=args.dec)

    if args.free_position:

        pts_model.pts.position.ra.free = True
        pts_model.pts.position.dec.free = True

    test_on_point_source(roi,
                         liff=args.liff,
                         point_source_model=pts_model,
                         maptree=args.maptree,
                         response=args.response)
