reaction
========

Channel topology and frame helpers.

Channels define particles as directed edges between interaction vertices. Declaring a
vertex creates the attached particles if they do not already exist, and particle
annotations can then be added with ``Channel.edit_particle``.

.. code-block:: python

   import laddu as ld

   channel = ld.Channel()
   channel.create_production("production", ["beam", "target"], ["rho", "spectator"])
   channel.create_decay("rho_decay", "rho", ["pi+", "pi-"])

   channel.edit_particle("beam", source=ld.ParticleSource.Stored, mass=0.0)
   channel.edit_particle("target", source=ld.ParticleSource.Missing, mass=0.938272)
   channel.edit_particle("spectator", source=ld.ParticleSource.Stored, mass=0.938272)
   channel.edit_particle("pi+", source=ld.ParticleSource.Stored, mass=0.13957)
   channel.edit_particle("pi-", source=ld.ParticleSource.Stored, mass=0.13957)

Frames are built explicitly from symbolic axes. The frame origin is the vertex whose
rest frame defines the measured angles. Each axis also declares the vertex where its
reference vector is evaluated.

.. code-block:: python

   frame = ld.Frame(
       "rho_decay",
       ld.Axes.from_y_z(
           ld.Axis.normal("beam", "spectator").at("production").flipped(),
           ld.Axis.opposite("spectator").at("rho_decay"),
       ),
   )

   angles = channel.angles("pi+", frame)
   costheta = angles.costheta
   phi = angles.phi

.. automodule:: laddu.reaction
   :members:
   :undoc-members:
