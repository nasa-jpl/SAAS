from poliastro.bodies import Venus, Mercury, Earth, Mars, Jupiter, Saturn, Uranus, Neptune, Body

body_mapping = {
    "earth": Earth,
    "mars": Mars,
    "venus": Venus,
    "jupiter": Jupiter,
    "mercury": Mercury,
    "saturn": Saturn,
    "uranus": Uranus,
    "neptune": Neptune,
}

def get_poliastro_body(body_name) -> Body:
    try:
        return body_mapping[body_name.lower()]
    except KeyError:
        raise ValueError(f"Invalid body name: '{body_name}'. Supported bodies: {list(body_mapping.keys())}")    
