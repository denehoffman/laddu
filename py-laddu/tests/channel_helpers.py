from __future__ import annotations

from collections.abc import Sequence

import laddu as ld


def channel(parent: str = 'kk') -> ld.Channel:
    ch = ld.Channel()
    ch.create_production('production', ['beam', 'target'], [parent, 'proton'])
    ch.create_decay(f'{parent}_decay', parent, ['kshort1', 'kshort2'])
    ch.edit_particle('beam', source=ld.ParticleSource.Stored)
    ch.edit_particle('target', source=ld.ParticleSource.Missing)
    ch.edit_particle('kshort1', source=ld.ParticleSource.Stored)
    ch.edit_particle('kshort2', source=ld.ParticleSource.Stored)
    ch.edit_particle('proton', source=ld.ParticleSource.Stored)
    return ch


def helicity_frame(parent: str = 'kk') -> ld.Frame:
    return ld.Frame(
        f'{parent}_decay',
        ld.Axes.from_y_z(
            ld.Axis.normal('beam', 'proton').at('production').flipped(),
            ld.Axis.opposite('proton').at(f'{parent}_decay'),
        ),
    )


def gottfried_jackson_frame(parent: str = 'kk') -> ld.Frame:
    return ld.Frame(
        f'{parent}_decay',
        ld.Axes.from_y_z(
            ld.Axis.normal('beam', 'proton').at('production').flipped(),
            ld.Axis.particle('beam').at(f'{parent}_decay'),
        ),
    )


def mass(particles: str | Sequence[str]) -> ld.Mass:
    names = [particles] if isinstance(particles, str) else list(particles)
    ch = ld.Channel()
    if len(names) == 1:
        ch.create_decay('mass_vertex', 'mass_parent', [names[0], 'mass_spectator'])
        ch.edit_particle(names[0], source=ld.ParticleSource.Stored)
        return ch.mass(names[0])
    if len(names) == 2:
        ch.create_decay('mass_vertex', 'mass_parent', [names[0], names[1]])
        ch.edit_particle(names[0], source=ld.ParticleSource.Stored)
        ch.edit_particle(names[1], source=ld.ParticleSource.Stored)
        return ch.mass('mass_parent')
    msg = 'test helper only supports one- and two-particle masses'
    raise ValueError(msg)
