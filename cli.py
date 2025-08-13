
import argparse, os, sys, json, numpy as np, pandas as pd, matplotlib.pyplot as plt
from .datacollapse import fit_data_collapse, fit_data_collapse_fse, fit_data_collapse_fse_robust, collapse_transform

def main(argv=None):
    p = argparse.ArgumentParser(description="Finite-size data-collapse")
    p.add_argument("--csv", required=True, help="Input CSV with columns L,U,Y[,sigma]")
    p.add_argument("--outdir", default=".", help="Output directory for plots")
    p.add_argument("--fse", action="store_true", help="Use finite-size correction")
    p.add_argument("--normalize", action="store_true", help="Normalize (1+bL^c) by (1+b L_ref^c)")
    p.add_argument("--L_ref", default="geom", help="Reference L for normalization: 'geom' or a number")
    p.add_argument("--Uc0", type=float, default=8.6); p.add_argument("--a0", type=float, default=2.0)
    p.add_argument("--b0", type=float, default=0.5); p.add_argument("--c0", type=float, default=-0.3)
    p.add_argument("--n_knots", type=int, default=12); p.add_argument("--lam", type=float, default=1e-3)
    p.add_argument("--n_boot", type=int, default=10)
    p.add_argument("--bounds", type=str, default="", help="JSON list, e.g. [[8.4,8.9],[0.2,4],[0,3],[-2,-0.01]]")
    # robust options
    p.add_argument("--robust", action="store_true", help="Robust grid over (b,c) + (Uc,a) optimize")
    p.add_argument("--b_grid", type=str, default="", help="e.g. 0:3:0.2 (start:stop:step)")
    p.add_argument("--c_grid", type=str, default="", help="-1:-0.05:0.05")
    args = p.parse_args(argv)

    df = pd.read_csv(args.csv)
    data = df[["L","U","Y"]].to_numpy(float)
    err = df["sigma"].to_numpy(float) if "sigma" in df.columns else None
    bounds = json.loads(args.bounds) if args.bounds else None

    # BEFORE
    plt.figure()
    for L in sorted(df["L"].unique()):
        sub = df[df["L"]==L]
        line, = plt.plot(sub["U"], sub["Y"], "-", lw=1.2, label=f"L={L}")
        if "sigma" in df.columns:
            plt.errorbar(sub["U"], sub["Y"], yerr=sub["sigma"], fmt="o", ms=3, capsize=2, elinewidth=1, color=line.get_color())
        else:
            plt.plot(sub["U"], sub["Y"], "o", ms=3, color=line.get_color())
    plt.xlabel("U"); plt.ylabel("Y"); plt.title("Raw curves")
    plt.grid(True, alpha=0.25); plt.legend()
    os.makedirs(args.outdir, exist_ok=True)
    plt.tight_layout(); plt.savefig(os.path.join(args.outdir, "before.png"), dpi=180); plt.close()

    # Fit
    if args.fse:
        if args.robust:
            def parse_grid(s):
                a,b,c = s.split(":"); a=float(a); b=float(b); c=float(c)
                n = int(round((b-a)/c))+1
                return np.linspace(a,b,n)
            b_grid = parse_grid(args.b_grid) if args.b_grid else np.linspace(0.0, 3.0, 16)
            c_grid = parse_grid(args.c_grid) if args.c_grid else np.linspace(-1.0, -0.05, 20)
            (params, errs) = fit_data_collapse_fse_robust(
                data, err, args.Uc0, args.a0, b_grid, c_grid,
                n_knots=args.n_knots, lam=args.lam, n_boot=args.n_boot,
                bounds_Ua=((args.Uc0-0.5, args.Uc0+0.5), (0.0, 4.0)),
                normalize=args.normalize, L_ref=args.L_ref
            )
        else:
            (params, errs) = fit_data_collapse_fse(
                data, err, args.Uc0, args.a0, args.b0, args.c0,
                n_knots=args.n_knots, lam=args.lam, n_boot=args.n_boot,
                bounds=bounds, normalize=args.normalize, L_ref=args.L_ref
            )
        x, Ycorr = collapse_transform(data, params, normalize=args.normalize, L_ref=args.L_ref)
    else:
        (params_na, errs_na) = fit_data_collapse(
            data, err, args.Uc0, args.a0, n_knots=args.n_knots, lam=args.lam, n_boot=args.n_boot,
            bounds=bounds
        )
        params = params_na; errs = errs_na
        x, Ycorr = collapse_transform(data, params)

    # AFTER
    plt.figure()
    for L in sorted(df["L"].unique()):
        m = (df["L"]==L).to_numpy()
        xs = x[m]; ys = Ycorr[m]
        if args.fse and "sigma" in df.columns:
            L_vals = df["L"][m].to_numpy(float)
            b = params[2] if len(params)>=3 else 0.0
            c = params[3] if len(params)>=4 else -0.2
            if args.normalize:
                # need to divide sigma by normalized scale too
                if args.L_ref == "geom":
                    Lr = float(np.exp(np.mean(np.log(df["L"].to_numpy(float)))))
                else:
                    Lr = float(args.L_ref)
                S = (1.0 + b*(L_vals**c)) / (1.0 + b*(Lr**c))
            else:
                S = (1.0 + b*(L_vals**c))
            ss = (df["sigma"][m].to_numpy() / S)
        else:
            ss = df["sigma"][m].to_numpy() if "sigma" in df.columns else None
        order = np.argsort(xs); xs, ys = xs[order], ys[order]
        line, = plt.plot(xs, ys, "-", lw=1.2, label=f"L={int(L)}")
        if ss is not None:
            ss = ss[order]
            plt.errorbar(xs, ys, yerr=ss, fmt="o", ms=3, capsize=2, elinewidth=1, color=line.get_color())
        else:
            plt.plot(xs, ys, "o", ms=3, color=line.get_color())
    plt.xlabel("(U - Uc) * L^a"); plt.ylabel("Y / (1 + b L^c)" if args.fse else "Y")
    title = f"Collapsed ({'FSE' if args.fse else 'no-FSE'}) params: " + ", ".join(f\"{v:.4g}\" for v in params)
    plt.title(title); plt.grid(True, alpha=0.25); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(args.outdir, "after.png"), dpi=180); plt.close()

    print("Params:", params, "+/-", errs)
    print("Saved plots to", os.path.abspath(args.outdir))

if __name__ == "__main__":
    main()
