Unbinned Fitting Tutorial
=========================

Theory
------

Perhaps the simplest kind of fitting we can do with ``laddu`` is one which involves fitting all of the data simultaneously to one model. Suppose we have some data representing events from a particular reaction. These data might contain resonances from intermediate particles which can be reconstructed from the four-momenta we've recorded, and those resonances have their own observable properties like angular momentum and parity. We can construct a model of these resonances in both mass :math:`m` and angular space :math:`\Omega` and define :math:`p(x; m, \Omega)` to be the probability that an event :math:`x` has the given phase space distribution. Since we also observe events themselves in a probabilistic manner, we must also consider the probability of observing :math:`N` events from such a process. This can be done with an extended maximum likelihood, following the derivation by [Barlow]_. First, we admit that while we defined :math:`p(x; m, \Omega)`, we assumed it would have unit normalization. However, we will now consider replacing this with a function whose normalization is not so constrained, called the intensity:

.. math:: \int \mathcal{I}(x; m, \Omega) \text{d}x = \mathcal{N}(m, \Omega)

We do this because our observed number of events :math:`N` will deviate from the expected number predicted by our model (:math:`\mathcal{N}(m, \Omega)`) according to Poisson statistics, since the observation of events themselves is a random process and are also subject to the efficiency of the detector (which we will discuss later).

Of course, we now have the problem of maximizing the resultant likelihood from this unnormalized distribution. We can write the likelihood using a Poisson probability distribution multiplied by the original product of probabilities over each observed event:

.. math:: 

   \mathcal{L} &= e^{-\mathcal{N}}\frac{\mathcal{N}^N}{N!} \prod_{i=1}^{N} p(x_i; m, \Omega) \\
   &= \frac{e^{-\mathcal{N}}}{N!} \prod_{i=1}^{N} \mathcal{I}(x_i; m, \Omega)

given that the extended probability is related to the standard one by :math:`\mathcal{I} = \mathcal{N} p`. Next, we will consider that the efficiency of the detector can be modeled with a function :math:`\eta(x)`. In reality, we generally cannot know this function to any level where it would be useful in such a minimization, but we can approximate it through a finite sum of simulated events passed through a simulation of the detector used in the experiment. We will then say that

.. math:: \mathcal{N}'(m,\Omega) = \int \mathcal{I}(x; m, \Omega)\eta(x)\text{d}x

gives the predicted number of events with efficiency encorporated, so

.. math:: \mathcal{L} = \frac{e^{-\mathcal{N}'}}{N!}\prod_{i=1}^{N}\mathcal{I}(x_i; m, \Omega)

While we mathematically could maximize the likelihood given above, a large product of terms between zero and one (or floating point values in general) is computationally unstable. Instead, we rephrase the problem from maximimizing the likelihood to maximizing the natural log of the likelihood, since the logarithm is monotonic and the log of a product is just the sum of the logs of the terms. Futhermore, since most optimization algorithms prefer to minimize functions rather than maximize them, we can just flip the sign. The the negative log of the extended likelihood (times two for error estimation purposes) is given by

.. math:: 

   -2\ln\mathcal{L} &= -2\left(\ln\left[\prod_{i=1}^{N}\mathcal{I}(x; m, \Omega)\right] - \mathcal{N}' + \ln N! \right) \\
   &= -2\left(\ln\left[\prod_{i=1}^{N}\mathcal{I}(x; m, \Omega)\right] - \left[\int \mathcal{I}(x; m, \Omega)\eta(x)\text{d}x \right] + \ln N! \right) \\
   &= -2\left(\left[\sum_{i=1}^{N}\ln \mathcal{I}(x; m, \Omega)\right] - \left[\int \mathcal{I}(x; m, \Omega)\eta(x)\text{d}x \right] + \ln N! \right)

As mentioned, we don't actually know the analytical form of :math:`\eta(x)`, but we can approximate it using Monte Carlo data. Assume we generate some data without any explicit physics model other than the phase space of the channel and pass it through a simulation of the detector. We will call these the "generated" and "accepted" datasets. We can approximate this integral a finite sum over this simulated data:

.. math:: \int \mathcal{I}(x; m, \Omega)\eta(x)\text{d}x \approx \frac{1}{\mathbb{P} N_g} \sum_{i=1}^{N_a} \mathcal{I}(x_i; m, \Omega)

where :math:`N_g` and :math:`N_a` are the size of the generated and accepted datasets respectively, the sum is over accepted events only, and :math:`\mathbb{P}` is the area of the integration region. This last term is another unknown, but in practice, we can consider that :math:`\mathcal{I}` could be rescaled by this factor, and that the multiplicative factor in the first part of the negative-log-likelihood would be extracted as the additive term :math:`N\ln\mathbb{P}` which is a constant in parameter space and therefore doesn't effect the overall minimization.

Removing all such constants, we obtain the following form for the negative log-likelihood:

.. math:: -2\ln\mathcal{L} = -2\left(\left[\sum_{i=1}^{N}\ln \mathcal{I}(x; m, \Omega)\right] - \left[ \frac{1}{N_g} \sum_{i=1}^{N_a} \mathcal{I}(x_i; m, \Omega) \right]\right)

Next, consider that events in both the data and in the Monte Carlo might have weights associated with them. We can easily adjust the negative log-likelihood to account for weighted events:

.. math:: -2\ln\mathcal{L} = -2\left(\left[\sum_{i=1}^{N} w_i \ln \mathcal{I}(x; m, \Omega)\right] - \left[ \frac{1}{N_g} \sum_{i=1}^{N_a} w_i \mathcal{I}(x_i; m, \Omega) \right]\right)

To visualize the result after minimization, we can weight each accepted Monte Carlo event by :math:`w \mathcal{L} / N_a` to see the result without acceptance correction, or we can weight each generated Monte Carlo event by :math:`\mathcal{L} / N_a` (generally the generated Monte Carlo is not weighted) to obtain the result corrected for the efficiency of the detector.

Example
-------

``laddu`` takes care of most of the math above and requires the user to provide data and the intensity function :math:`I(\text{event};\text{parameters})`. For the rest of this tutorial, this function will be refered to as the "model". In ``laddu``, we construct the model out of modular "amplitudes" (:math:`A(e; \vec{p})`) which can be added and multiplied together. Additionally, since these amplitudes are functions on the space :math:`\mathbb{R}^n \to \mathbb{C}`, we can also take their real part, imaginary part, or the square of their norm (:math:`AA^*`). Generally, when building a model, we compartmentalize amplitude terms into coherent summations, where coherent just means we take the norm-squared of the sum. Since the likelihood should be strictly real, the real part of the model's evaluation is taken to calculate the intensity, so imaginary terms (which should be zero in practice) are discarded.

For a simple unbinned fit, we must first obtain some data. ``laddu`` does not currently have a built-in event generator, so it is recommended that users utilize other methods of generating Monte Carlo data. For the sake of this tutorial, we will assume that these data files are readily available as Parquet files.

.. note:: The Parquet file format is not common to particle physics but is ubiquitous in data science. The structure that ``laddu`` requires is specified in the API reference and can be generated via `pandas <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_parquet.html>`_, `polars <https://docs.pola.rs/api/python/stable/reference/api/polars.DataFrame.write_parquet.html>`_ or most other data libraries. The only difficulty is translating existing data (which is likely in the ROOT format) into this representation. For this process, `uproot <https://uproot.readthedocs.io/en/latest/>`_ is recommended to avoid using ROOT directly. There is also an executable ``amptools-to-laddu`` which is installed alongside the Python package which can convert directly from ROOT files in the AmpTools format to the equivalent ``laddu`` Parquet files. The Python API also exposes the underlying conversion method in its ``convert`` submodule.

Reading data with ``laddu`` is as simple as using the `laddu.open` method. It takes the path to the data file as its argument:

.. code-block:: python

   import laddu as ld

   data_ds = ld.open("data.parquet")
   accmc_ds = ld.open("accmc.parquet")
   genmc_ds = ld.open("genmc.parquet")

Next, we need to construct a model. Let's assume that the dataset contains events from the channel :math:`\gamma p \to K_S^0 K_S^0 p'` and that the measured particles in the data files are :math:`[\gamma, p', K_{S,1}^0, K_{S,2}^0]`. This setup mimics the GlueX experiment at Jefferson Lab (the momentum of the initial proton target is not measured and can be reasonably assumed to be close to zero in magnitude). 

.. note:: The four-momenta in the datasets need to be in the center-of-momentum frame, which is the only frame that can be considered invariant between different experiments. Some of the amplitudes used will boost particles from the center-of-momentum frame to some new frame, and this is a distinct transformation from boosting directly from a lab frame to the same target frame!

Let's further assume that there are only two resonances present in our data, an :math:`f_0(1500)` and a :math:`f_2'(1525)` [#f1]_. We will assume that the data were generated via two relativistic Breit-Wigner distributions with masses at :math:`1506\text{ MeV}/c^2` and :math:`1517\text{ MeV}/c^2` respectively and widths of :math:`112\text{ MeV}/c^2` and :math:`86\text{ MeV}/c^2` respectively (these values come from the PDG). These resonances also have spin, so we can look at their decay angles as well as the overall mass distribution. These variables are all defined by ``laddu`` as helper classes:

.. code:: python

   # the mass of the combination of particles 2 and 3, the kaons
   res_mass = ld.Mass([2, 3])

   # the decay angles in the helicity frame
   angles = ld.Angles(0, [1], [2], [2, 3])

So far, these angles just represent particles in a generic dataset by index and provide an appropriate method to calculate the corresponding observable. Before we fit anything, we might want to just see what the dataset looks like:

.. code:: python

   import matplotlib.pyplot as plt

   m_data = res_mass.value_on(data_ds)
   costheta_data = angles.costheta.value_on(data_ds)
   phi_data = angles.phi.value_on(data_ds)

   fig, ax = plt.subplots(ncols=2)
   ax[0].hist(m_data, bins=100)
   ax[0].set_xlabel('Mass of $K_SK_S$ in GeV/$c^2$')
   ax[1].hist2d(costheta_data, phi_data, bins=(100, 100))
   ax[1].set_xlabel('$\cos(\theta_{HX})$')
   ax[1].set_ylabel('$\varphi_{HX}$')
   plt.tight_layout()
   plt.show()



.. rubric:: Footnotes

.. [#f1] In reality, there are many more resonances present in this channel, and the model we are about to construct technically doesn't preserve unitarity, but this is just a simple example to demonstrate the mechanics of ``laddu``.

.. [Barlow] Barlow, R. (1990). Extended maximum likelihood. Nuclear Instruments and Methods in Physics Research Section A: Accelerators, Spectrometers, Detectors and Associated Equipment, 297(3), 496–506. doi:10.1016/0168-9002(90)91334-8
