# Task 3: Lockfile and reproducibility
# TODO: Generate uv.lock
# TODO: Explain in 3-5 comments why lockfiles matter for teams

# 1) Same deps for everyone: a lockfile pins exact versions so “works on my machine” stops being a personality trait.
# 2) Reproducible CI/CD: your tests/builds run against the same dependency set locally and in CI, reducing flaky failures.
# 3) Safer upgrades: changes become explicit diffs in the lockfile, making dependency updates reviewable and reversible.
# 4) Faster installs: lockfiles let tools resolve once and install many times, which speeds up onboarding and pipelines.
# 5) Better debugging: when something breaks, you can tie it to a specific dependency change instead of guessing.
