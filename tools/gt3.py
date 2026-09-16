
import binascii
for l in open("pyproject.toml").read().splitlines():
    low = l.lower()
    if any(k in low.replace(" ", "") for k in ("build-backend", "pydantic", "license", "hatchl", "requires-python")):
        print(binascii.hexlify(l.encode()).decode())
