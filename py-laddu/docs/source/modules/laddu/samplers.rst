samplers
===

``laddu.samplers`` contains small constructors used to annotate channels for event
generation. These helpers configure particle momentum sources, particle mass
samplers, and generated two-to-two production vertices.

Mass samplers default to using ``ParticleProperties``. Use an explicit sampler
only for generated resonances or other particles whose mass should be sampled.

.. code-block:: python

   channel.edit_particle("beam", mass=0.0, momentum=ld.samplers.energy(8.0))
   channel.edit_particle("target", mass=0.938272, momentum=ld.samplers.rest())
   channel.edit_particle("rho", mass_sampler=ld.samplers.uniform_mass(0.6, 0.9))

   channel.edit_vertex("production", generator=ld.samplers.t_exponential(0.1))

.. automodule:: laddu.samplers
   :members:
   :undoc-members:
