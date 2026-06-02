generation
==========

``laddu.generation`` contains event generator objects and validated generation
plans. Channel annotation constructors live in :mod:`laddu.gen`.

.. code-block:: python

   import laddu as ld
   from laddu import generation

   channel = ld.Channel()
   channel.create_production(
       "production",
       ["beam", "target"],
       ["rho", "spectator"],
       generator=ld.gen.t_exponential(0.1),
   )
   channel.create_decay("rho_decay", "rho", ["pi+", "pi-"])

   channel.edit_particle("beam", mass=0.0, momentum=ld.gen.energy(8.0))
   channel.edit_particle("target", mass=0.938272, momentum=ld.gen.rest())
   channel.edit_particle("rho", mass_sampler=ld.gen.uniform_mass(0.6, 0.9))
   channel.edit_particle("spectator", mass=0.938272)
   channel.edit_particle("pi+", mass=0.13957)
   channel.edit_particle("pi-", mass=0.13957)

   generator = generation.EventGenerator(channel, seed=12345)
   dataset = generator.generate_dataset(1000)

.. automodule:: laddu.generation
   :members:
   :undoc-members:
