"""
Curated destinations.

Every one of these was found by the image-driven descent in deepzoom.py rather
than typed from a list somewhere: render a coarse grid, aim at the pixel that
escaped with the highest iteration count, shrink, repeat. Each was then
rendered and looked at, because an automated deep-zoom search will happily
converge on a featureless region and report success.

Coordinates carry far more digits than float64 can hold. That is the point --
they are parsed at arbitrary precision for the reference orbit.
"""

LOCATIONS = [
    {
        "name": 'Seahorse Valley',
        "cx": '-0.7429228277490906641340983480246393518594381442325666',
        "cy": '0.1312172046120566657097074968140034621626924517673668',
        "radius": 3.2311742677852644e-29,
        "maxiter": 22022,
    },
    {
        "name": 'Elephant Valley',
        "cx": '0.29304074444829339982583304878234978770339355233470',
        "cy": '0.01499668258898927982531512648051669801761493134153',
        "radius": 8.271806125530277e-27,
        "maxiter": 20336,
    },
    {
        "name": 'Triple Spiral',
        "cx": '-0.088426893634964119765601963048148739363899199793',
        "cy": '0.655358807121304543853646169056162984704894176479',
        "radius": 5.293955920339377e-25,
        "maxiter": 19071,
    },
    {
        "name": 'Scepter Valley',
        "cx": '-1.749193532808635129204146027605135429375700047004488201',
        "cy": '0.000014185319660156331139814651795146410547269047617390',
        "radius": 3.2311742677852645e-31,
        "maxiter": 23422,
    },
    {
        "name": 'North Antenna',
        "cx": '-0.10109638498625052817737701644213508313538485560704',
        "cy": '0.95628651119574910047341614268654044653843181493632',
        "radius": 3.308722450212111e-27,
        "maxiter": 20614,
    },
    {
        "name": 'Deep Filament',
        "cx": '-1.25068094110280061607722374040737274194522710326963895080',
        "cy": '0.02011943766084348810796807733457020103968014973511666269',
        "radius": 3.155443620884047e-33,
        "maxiter": 24829,
    },
]


HOME = {
    "name": "The Whole Set",
    "cx": "-0.5",
    "cy": "0.0",
    "radius": 1.6,
    "maxiter": 700,
}


def by_index(i):
    return LOCATIONS[i % len(LOCATIONS)]
