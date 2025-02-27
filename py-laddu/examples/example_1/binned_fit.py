import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import laddu as ld

res_mass = ld.Mass([2, 3])
bins = 50
mass_range = (1.0, 2.0)

data_ds = ld.open('data_1.parquet')
accmc_ds = ld.open('accmc_1.parquet')
# We'll want to retain the unbinned datasets for plotting later

binned_data_ds = data_ds.bin_by(res_mass, bins, mass_range)
binned_accmc_ds = accmc_ds.bin_by(res_mass, bins, mass_range)

manager = ld.Manager()
angles = ld.Angles(0, [1], [2], [2, 3])
polarization = ld.Polarization(0, [1], 0)

z00p = manager.register(ld.Zlm('Z00+', 0, 0, '+', angles, polarization))
z22p = manager.register(ld.Zlm('Z22+', 2, 2, '+', angles, polarization))

s0p = manager.register(ld.Scalar('S0+', ld.parameter('Re[S0+]')))
d2p = manager.register(
    ld.ComplexScalar('D2+', ld.parameter('Re[D2+]'), ld.parameter('Im[D2+]'))
)

positive_real_sum = (s0p * z00p.real() + d2p * z22p.real()).norm_sqr()
positive_imag_sum = (s0p * z00p.imag() + d2p * z22p.imag()).norm_sqr()
model = manager.model(positive_real_sum + positive_imag_sum)

nlls = [ld.NLL(model, binned_data_ds[i], binned_accmc_ds[i]) for i in range(bins)]

binned_statuses = []
for nll in nlls:
    binned_statuses.append(nll.minimize([100.0, 100.0, 100.0]))

# 'wb' for *binary* writing mode
with Path('binned_statuses.pkl').open('wb') as output_path:
    pickle.dump(binned_statuses, output_path)
# We can load this up later with
# binned_statuses = pickle.load(Path("binned_statuses.pkl").open("rb"))


# sorted_accmc_ds = sum([ds for ds in binned_accmc_ds])
# assert sorted_accmc_ds != 0  # this line is for type-checkers only!
# tot_weights = np.concatenate(
#     [np.array(nlls[ibin].project(binned_statuses[ibin].x)) for ibin in range(bins)]
# )
# s0p_weights = np.concatenate(
#     [
#         np.array(nlls[ibin].project_with(binned_statuses[ibin].x, ['S0+', 'Z00+']))
#         for ibin in range(bins)
#     ]
# )
# d2p_weights = np.concatenate(
#     [
#         np.array(nlls[ibin].project_with(binned_statuses[ibin].x, ['D2+', 'Z22+']))
#         for ibin in range(bins)
#     ]
# )

edges = np.histogram_bin_edges([], bins, mass_range)

tot_counts = []
s0p_counts = []
d2p_counts = []

for ibin in range(bins):
    tot_counts.append(nlls[ibin].project(binned_statuses[ibin].x).sum())
    s0p_counts.append(
        nlls[ibin].project_with(binned_statuses[ibin].x, ['S0+', 'Z00+']).sum()
    )
    d2p_counts.append(
        nlls[ibin].project_with(binned_statuses[ibin].x, ['D2+', 'Z22+']).sum()
    )


fig, ax = plt.subplots(ncols=2, sharey=True)

# Plot the data on both axes
m_data = res_mass.value_on(data_ds)
ax[0].hist(
    m_data,
    bins=bins,
    range=mass_range,
    weights=data_ds.weights,
    color='k',
    histtype='step',
    label='Data',
)
ax[1].hist(
    m_data,
    bins=bins,
    range=mass_range,
    weights=data_ds.weights,
    color='k',
    histtype='step',
    label='Data',
)

# Plot the total fit on both axes
ax[0].stairs(tot_counts, edges, color='k', alpha=0.1, fill=True, label='Fit')
ax[1].stairs(tot_counts, edges, color='k', alpha=0.1, fill=True, label='Fit')

# Plot the S-wave on the left
ax[0].stairs(s0p_counts, edges, color='r', alpha=0.1, fill=True, label='$S_0^{(+)}$')

# Plot the D-wave on the right
ax[1].stairs(d2p_counts, edges, color='r', alpha=0.1, fill=True, label='$D_2^{(+)}$')

ax[0].legend()
ax[1].legend()
ax[0].set_ylim(0)
ax[1].set_ylim(0)
ax[0].set_xlabel('Mass of $K_S^0 K_S^0$ (GeV/$c^2$)')
ax[1].set_xlabel('Mass of $K_S^0 K_S^0$ (GeV/$c^2$)')
ax[0].set_ylabel('Counts / 10 MeV/$c^2$')
ax[1].set_ylabel('Counts / 10 MeV/$c^2$')
plt.tight_layout()
plt.savefig('binned_fit_result.png')
plt.close()

sd_phase = []
for ibin in range(bins):
    s = binned_statuses[ibin].x[0] + 0j
    d = binned_statuses[ibin].x[1] + 1j * binned_statuses[ibin].x[2]
    phase = abs(np.angle(s) - np.angle(d))
    sd_phase.append(min(phase, 2 * np.pi - phase))

fig, ax = plt.subplots()
ax.hist(
    m_data,
    bins=bins,
    range=mass_range,
    weights=data_ds.weights,
    color='k',
    histtype='step',
    label='Data',
)
phase_ax = ax.twinx()
centers = (edges[1:] + edges[:-1]) / 2
phase_ax.plot(centers, sd_phase, color='m', label='$S-D$ Phase')

ax.legend()
phase_ax.legend()
ax.set_ylim(0)
ax.set_xlabel('Mass of $K_S^0 K_S^0$ (GeV/$c^2$)')
ax.set_ylabel('Counts / 10 MeV/$c^2$')
phase_ax.set_ylabel('Angle (rad)')
phase_ax.spines['right'].set_color('m')
phase_ax.tick_params(axis='y', colors='m')
plt.tight_layout()
plt.savefig('binned_fit_result_phase.png')
plt.close()
