# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "corner",
#     "laddu",
#     "loguru",
#     "matplotlib",
#     "numpy",
# ]
# ///
from __future__ import annotations

import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from corner import corner
from loguru import logger

import laddu as ld


# This custom terminator differs from the one provided by `laddu`. Rather than tracing the
# walker positions and calculating the IAT, this first projects the current walker positions
# onto the two constituent waves and uses those to calculate a different IAT. This converges
# better because there is an implicit symmetry in the problem (only the absolute phase between
# the waves matters, not the sign of that phase) so walkers can bounce between two equivalent
# positions in the fit space which are very separate in the parameter space. It also
# demonstrates how to write and use a custom terminator.
class CustomAutocorrelationTerminator(ld.MCMCTerminator):
    def __init__(
        self,
        nll: ld.NLL,
        ncheck: int = 20,
        dact: float = 0.05,
        nact: int = 20,
        discard: float = 0.5,
    ) -> None:
        self.nll = nll
        self.ncheck = ncheck
        self.dact = dact
        self.nact = nact
        self.discard = discard
        self.latest_tau = np.inf
        self.tot = []
        self.s0s = []
        self.d2s = []

    def check_for_termination(
        self, step: int, status: ld.EnsembleStatus
    ) -> ld.ControlFlow:
        latest_step = status.get_chain()[:, -1, :]
        tot = []
        s0s = []
        d2s = []
        for i_walker in range(status.dimension[0]):
            tot.append(np.sum(self.nll.project(latest_step[i_walker])))
            s0s.append(
                np.sum(self.nll.project_with(latest_step[i_walker], ['Z00+', 'S0+']))
            )
            d2s.append(
                np.sum(self.nll.project_with(latest_step[i_walker], ['Z22+', 'D2+']))
            )
        self.tot.append(tot)
        self.s0s.append(s0s)
        self.d2s.append(d2s)
        if step % self.ncheck == 0:
            logger.info('Checking Autocorrelation (custom)')
            logger.info(
                f'Chain dimensions: {status.dimension[0]} walkers, {status.dimension[1]} steps, {status.dimension[2]} parameters'
            )
            chain = np.array([self.s0s, self.d2s]).transpose(
                2, 1, 0
            )  # (walkers, steps, parameters)
            chain = chain[
                :,
                min(
                    int(step * self.discard),
                    int(self.latest_tau * self.nact)
                    if np.isfinite(self.latest_tau)
                    else int(step * self.discard),
                ) :,
            ]
            taus = ld.integrated_autocorrelation_times(chain)
            logger.info(f'τ = [{", ".join(str(t) for t in taus)}]')
            tau = np.mean(taus)
            logger.info(f'mean τ = {tau}')
            logger.info(f'steps to converge = {int(tau * self.nact)}')
            logger.info(f'steps remaining = {int(tau * self.nact) - step}')
            logger.info(
                f'Δτ/τ = {abs(self.latest_tau - tau) / tau} (converges if < {self.dact})'
            )
            logger.info('End of custom Autocorrelation check')
            converged = (tau * self.nact < step) and (
                abs(self.latest_tau - tau) / tau < self.dact
            )
            self.latest_tau = tau
            if converged:
                return ld.ControlFlow.Break
        return ld.ControlFlow.Continue


def main() -> None:
    script_dir = Path(os.path.realpath(__file__)).parent.resolve()
    data_dir = script_dir.parent / 'data'
    data_file = str(data_dir / 'data.parquet')
    accmc_file = str(data_dir / 'accmc.parquet')
    p4_columns = ['beam', 'proton', 'kshort1', 'kshort2']
    aux_columns = ['pol_magnitude', 'pol_angle']
    logger.info('Opening Data file...')
    data_ds = ld.Dataset.from_parquet(data_file, p4s=p4_columns, aux=aux_columns)
    logger.info('Opening AccMC file...')
    accmc_ds = ld.Dataset.from_parquet(accmc_file, p4s=p4_columns, aux=aux_columns)

    res_mass = ld.Mass(['kshort1', 'kshort2'])
    m_data = res_mass.value_on(data_ds)

    font = {'family': 'DejaVu Sans', 'weight': 'normal', 'size': 22}
    plt.rc('font', **font)
    plt.rc('axes', titlesize=48)
    plt.rc('legend', fontsize=24)
    plt.rc('xtick', direction='in')
    plt.rc('ytick', direction='in')
    plt.rcParams['xtick.minor.visible'] = True
    plt.rcParams['xtick.minor.size'] = 4
    plt.rcParams['xtick.minor.width'] = 1
    plt.rcParams['xtick.major.size'] = 8
    plt.rcParams['xtick.major.width'] = 1
    plt.rcParams['ytick.minor.visible'] = True
    plt.rcParams['ytick.minor.size'] = 4
    plt.rcParams['ytick.minor.width'] = 1
    plt.rcParams['ytick.major.size'] = 8
    plt.rcParams['ytick.major.width'] = 1

    red = '#EF3A47'
    orange = '#FF7F00'
    blue = '#007BC0'
    purple = '#984EA3'
    black = '#000000'

    with Path('example_1_binned_fit.pkl').open('rb') as binned_fit_file:
        binned_fits = pickle.load(binned_fit_file)

    bins = binned_fits['nbins']

    data_ds_binned = data_ds.bin_by(res_mass, bins, binned_fits['range'])
    accmc_ds_binned = accmc_ds.bin_by(res_mass, bins, binned_fits['range'])

    bin_edges = data_ds_binned.edges

    tots = []
    s0p = []
    d2p = []
    tot_err_boot_lower = []
    tot_err_boot_upper = []
    s0p_err_boot_lower = []
    s0p_err_boot_upper = []
    d2p_err_boot_lower = []
    d2p_err_boot_upper = []
    tot_err_mcmc_lower = []
    tot_err_mcmc_upper = []
    s0p_err_mcmc_lower = []
    s0p_err_mcmc_upper = []
    d2p_err_mcmc_lower = []
    d2p_err_mcmc_upper = []
    for ibin in range(binned_fits['nbins']):
        fit = binned_fits['fits'][ibin]
        model = fit['model']
        best = fit['best']
        bootstraps = fit['bootstraps']
        nll = ld.NLL(model, data_ds_binned[ibin], accmc_ds_binned[ibin])
        tot_fit = np.sum(nll.project(best.x))
        s0_fit = np.sum(nll.project_with(best.x, ['Z00+', 'S0+']))
        d2_fit = np.sum(nll.project_with(best.x, ['Z22+', 'D2+']))
        tot_boot = []
        s0_boot = []
        d2_boot = []
        for bootstrap in bootstraps:
            tot_boot.append(np.sum(nll.project(best.x)))
            s0_boot.append(np.sum(nll.project_with(bootstrap.x, ['Z00+', 'S0+'])))
            d2_boot.append(np.sum(nll.project_with(bootstrap.x, ['Z22+', 'D2+'])))

        tot_ci_boot = np.quantile(tot_boot, [0.16, 0.84])
        s0s_ci_boot = np.quantile(s0_boot, [0.16, 0.84])
        d2s_ci_boot = np.quantile(d2_boot, [0.16, 0.84])

        output_path = Path(f'bin_{ibin}_mcmc.pkl')
        if output_path.exists():
            with output_path.open('rb') as bin_out_file:
                bin_out = pickle.load(bin_out_file)
                ensemble = bin_out['ensemble']
                tau = bin_out['tau']
                taus = bin_out['taus']
            (_, n_steps, _) = ensemble.dimension
            n_steps_burned = n_steps - int(tau * 10)  # 210
            requested_steps = 100
            excess_steps = n_steps_burned - requested_steps  # 110
            thin = 1 if excess_steps < 0 else n_steps_burned // requested_steps
            flat_chain = ensemble.get_flat_chain(burn=int(tau * 3), thin=thin)
            tot = []
            s0s = []
            d2s = []
            for position in flat_chain:
                tot.append(np.sum(nll.project(position)))
                s0s.append(np.sum(nll.project_with(position, ['Z00+', 'S0+'])))
                d2s.append(np.sum(nll.project_with(position, ['Z22+', 'D2+'])))
        else:
            p0 = best.x + np.random.normal(0, scale=0.01, size=(100, len(best.x)))
            nll_clone = nll
            caco = CustomAutocorrelationTerminator(nll_clone)
            aco = ld.AutocorrelationTerminator(n_check=10, terminate=False, verbose=True)
            ensemble = nll.mcmc(
                p0, method='ess', max_steps=30000, terminators=[caco, aco]
            )
            tau = caco.latest_tau
            taus = aco.taus
            bin_out = {'ensemble': ensemble, 'tau': tau, 'taus': taus}
            tot = np.array(caco.tot).reshape(-1)
            s0s = np.array(caco.s0s).reshape(-1)
            d2s = np.array(caco.d2s).reshape(-1)
            (_, n_steps, _) = ensemble.dimension
            n_steps_burned = n_steps - int(tau * 10)  # 210
            requested_steps = 100
            excess_steps = n_steps_burned - requested_steps  # 110
            thin = 1 if excess_steps < 0 else n_steps_burned // requested_steps
            flat_chain = ensemble.get_flat_chain(burn=int(tau * 3), thin=thin)
            with Path(f'bin_{ibin}_mcmc.pkl').open('wb') as bin_out_file:
                pickle.dump(bin_out, bin_out_file)

        chain = ensemble.get_chain(burn=int(tau * 10), thin=thin).transpose(1, 0, 2)
        _, axes = plt.subplots(3, figsize=(10, 7), sharex=True)
        labels = ['$S_0^+$ real', '$D_2^+$ real', '$D_2^+$ imag']
        for i in range(3):
            ax = axes[i]
            ax.plot(np.arange(len(chain)) + int(tau * 10), chain[:, :, i], 'k', alpha=0.3)
            ax.set_xlim(int(tau * 10), len(chain) + int(tau * 10))
            ax.set_ylabel(labels[i])
            ax.yaxis.set_label_coords(-0.1, 0.5)
            axes[-1].set_xlabel('step number')
        plt.savefig(f'mcmc_plots/trace_{ibin}.svg')
        plt.close()
        plt.plot(np.arange(len(taus)) * 10, taus)
        plt.xlabel('Step')
        plt.ylabel(r'Mean $\tau$')
        plt.tight_layout()
        plt.savefig(f'mcmc_plots/iat_{ibin}.svg')
        plt.close()
        corner(
            flat_chain,
            truths=best.x,
            quantiles=[0.16, 0.5, 0.84],
            labels=labels,
            show_titles=True,
            title_kwargs={'fontsize': 12},
        )
        plt.savefig(f'mcmc_plots/corner_{ibin}.svg')
        plt.close()
        corner(
            np.array([s0s, d2s]).transpose(),
            truths=[s0_fit, d2_fit],
            quantiles=[0.16, 0.5, 0.84],
            labels=['$S_0^+$', '$D_2^+$'],
            show_titles=True,
            title_kwargs={'fontsize': 12},
        )
        plt.savefig(f'mcmc_plots/corner_transformed_{ibin}.svg')
        plt.close()
        tot_ci_mcmc = np.quantile(tot, [0.16, 0.84])
        s0s_ci_mcmc = np.quantile(s0s, [0.16, 0.84])
        d2s_ci_mcmc = np.quantile(d2s, [0.16, 0.84])
        logger.info(
            f'Total (bootstrap) = {tot_fit} + {tot_ci_boot[1] - tot_fit} - {tot_fit - tot_ci_boot[0]}'
        )
        logger.info(
            f'S0+ (bootstrap) = {s0_fit} + {s0s_ci_boot[1] - s0_fit} - {s0_fit - s0s_ci_boot[0]}'
        )
        logger.info(
            f'D2+ (bootstrap) = {d2_fit} + {d2s_ci_boot[1] - d2_fit} - {d2_fit - d2s_ci_boot[0]}'
        )
        logger.info(
            f'Total (MCMC) = {tot_fit} + {tot_ci_mcmc[1] - tot_fit} - {tot_fit - tot_ci_mcmc[0]}'
        )
        logger.info(
            f'S0+ (MCMC) = {s0_fit} + {s0s_ci_mcmc[1] - s0_fit} - {s0_fit - s0s_ci_mcmc[0]}'
        )
        logger.info(
            f'D2+ (MCMC) = {d2_fit} + {d2s_ci_mcmc[1] - d2_fit} - {d2_fit - d2s_ci_mcmc[0]}'
        )
        tots.append(tot_fit)
        s0p.append(s0_fit)
        d2p.append(d2_fit)
        tot_err_boot_lower.append(tot_ci_boot[0])
        tot_err_boot_upper.append(tot_ci_boot[1])
        s0p_err_boot_lower.append(s0s_ci_boot[0])
        s0p_err_boot_upper.append(s0s_ci_boot[1])
        d2p_err_boot_lower.append(d2s_ci_boot[0])
        d2p_err_boot_upper.append(d2s_ci_boot[1])
        tot_err_mcmc_lower.append(tot_ci_mcmc[0])
        tot_err_mcmc_upper.append(tot_ci_mcmc[1])
        s0p_err_mcmc_lower.append(s0s_ci_mcmc[0])
        s0p_err_mcmc_upper.append(s0s_ci_mcmc[1])
        d2p_err_mcmc_lower.append(d2s_ci_mcmc[0])
        d2p_err_mcmc_upper.append(d2s_ci_mcmc[1])
    tots = np.array(tots)
    s0p = np.array(s0p)
    d2p = np.array(d2p)
    tot_err_boot_lower = np.array(tot_err_boot_lower)
    tot_err_boot_upper = np.array(tot_err_boot_upper)
    s0p_err_boot_lower = np.array(s0p_err_boot_lower)
    s0p_err_boot_upper = np.array(s0p_err_boot_upper)
    d2p_err_boot_lower = np.array(d2p_err_boot_lower)
    d2p_err_boot_upper = np.array(d2p_err_boot_upper)
    tot_err_mcmc_lower = np.array(tot_err_mcmc_lower)
    tot_err_mcmc_upper = np.array(tot_err_mcmc_upper)
    s0p_err_mcmc_lower = np.array(s0p_err_mcmc_lower)
    s0p_err_mcmc_upper = np.array(s0p_err_mcmc_upper)
    d2p_err_mcmc_lower = np.array(d2p_err_mcmc_lower)
    d2p_err_mcmc_upper = np.array(d2p_err_mcmc_upper)

    _, ax = plt.subplots(ncols=2, sharey=True, figsize=(22, 12))
    ax[0].hist(
        m_data, bins=bins, range=(1, 2), color=black, histtype='step', label='Data'
    )
    ax[1].hist(
        m_data, bins=bins, range=(1, 2), color=black, histtype='step', label='Data'
    )
    centers = np.array((bin_edges[1:] + bin_edges[:-1]) / 2)
    bin_width = np.diff(bin_edges)[0]
    ax[0].scatter(
        centers - bin_width / 4, tots, marker='.', color=black, label='Fit Total'
    )
    ax[0].errorbar(
        centers - bin_width / 4,
        (tot_err_boot_lower + tot_err_boot_upper) / 2,
        yerr=[
            (tot_err_boot_lower + tot_err_boot_upper) / 2 - tot_err_boot_lower,
            tot_err_boot_upper - (tot_err_boot_lower + tot_err_boot_upper) / 2,
        ],
        fmt=' ',
        color=black,
        label='bootstrap errors',
    )
    ax[0].scatter(
        centers + bin_width / 4, tots, marker='.', color=black, label='Fit Total'
    )
    ax[0].errorbar(
        centers + bin_width / 4,
        (tot_err_mcmc_lower + tot_err_mcmc_upper) / 2,
        yerr=[
            (tot_err_mcmc_lower + tot_err_mcmc_upper) / 2 - tot_err_mcmc_lower,
            tot_err_mcmc_upper - (tot_err_mcmc_lower + tot_err_mcmc_upper) / 2,
        ],
        fmt=' ',
        color=black,
        label='MCMC errors',
    )
    ax[1].scatter(
        centers - bin_width / 4, tots, marker='.', color=black, label='Fit Total'
    )
    ax[1].errorbar(
        centers - bin_width / 4,
        (tot_err_boot_lower + tot_err_boot_upper) / 2,
        yerr=[
            (tot_err_boot_lower + tot_err_boot_upper) / 2 - tot_err_boot_lower,
            tot_err_boot_upper - (tot_err_boot_lower + tot_err_boot_upper) / 2,
        ],
        fmt=' ',
        color=black,
        label='bootstrap errors',
    )
    ax[1].scatter(
        centers + bin_width / 4, tots, marker='.', color=black, label='Fit Total'
    )
    ax[1].errorbar(
        centers + bin_width / 4,
        (tot_err_mcmc_lower + tot_err_mcmc_upper) / 2,
        yerr=[
            (tot_err_mcmc_lower + tot_err_mcmc_upper) / 2 - tot_err_mcmc_lower,
            tot_err_mcmc_upper - (tot_err_mcmc_lower + tot_err_mcmc_upper) / 2,
        ],
        fmt=' ',
        color=black,
        label='MCMC errors',
    )

    ax[0].scatter(centers - bin_width / 4, s0p, marker='.', color=blue, label='$S_0^+$')
    ax[0].errorbar(
        centers - bin_width / 4,
        (s0p_err_boot_lower + s0p_err_boot_upper) / 2,
        yerr=[
            (s0p_err_boot_lower + s0p_err_boot_upper) / 2 - s0p_err_boot_lower,
            s0p_err_boot_upper - (s0p_err_boot_lower + s0p_err_boot_upper) / 2,
        ],
        fmt=' ',
        color=blue,
        label='bootstrap errors',
    )
    ax[0].scatter(centers + bin_width / 4, s0p, marker='.', color=purple, label='$S_0^+$')
    ax[0].errorbar(
        centers + bin_width / 4,
        (s0p_err_mcmc_lower + s0p_err_mcmc_upper) / 2,
        yerr=[
            (s0p_err_mcmc_lower + s0p_err_mcmc_upper) / 2 - s0p_err_mcmc_lower,
            s0p_err_mcmc_upper - (s0p_err_mcmc_lower + s0p_err_mcmc_upper) / 2,
        ],
        fmt=' ',
        color=purple,
        label='MCMC errors',
    )

    ax[1].scatter(centers - bin_width / 4, d2p, marker='.', color=red, label='$D_2^+$')
    ax[1].errorbar(
        centers - bin_width / 4,
        (d2p_err_boot_lower + d2p_err_boot_upper) / 2,
        yerr=[
            (d2p_err_boot_lower + d2p_err_boot_upper) / 2 - d2p_err_boot_lower,
            d2p_err_boot_upper - (d2p_err_boot_lower + d2p_err_boot_upper) / 2,
        ],
        fmt=' ',
        color=red,
        label='bootstrap errors',
    )
    ax[1].scatter(centers + bin_width / 4, d2p, marker='.', color=orange, label='$D_2^+$')
    ax[1].errorbar(
        centers + bin_width / 4,
        (d2p_err_mcmc_lower + d2p_err_mcmc_upper) / 2,
        yerr=[
            (d2p_err_mcmc_lower + d2p_err_mcmc_upper) / 2 - d2p_err_mcmc_lower,
            d2p_err_mcmc_upper - (d2p_err_mcmc_lower + d2p_err_mcmc_upper) / 2,
        ],
        fmt=' ',
        color=orange,
        label='MCMC errors',
    )

    ax[0].legend()
    ax[1].legend()
    ax[0].set_ylim(0)
    ax[1].set_ylim(0)
    ax[0].set_xlabel('Mass of $K_S^0 K_S^0$ (GeV/$c^2$)')
    ax[1].set_xlabel('Mass of $K_S^0 K_S^0$ (GeV/$c^2$)')
    bin_width = int(1000 / bins)
    ax[0].set_ylabel(f'Counts / {bin_width} MeV/$c^2$')
    ax[1].set_ylabel(f'Counts / {bin_width} MeV/$c^2$')
    plt.tight_layout()
    plt.savefig('example_1_mcmc_errors.svg')
    plt.close()


if __name__ == '__main__':
    main()
