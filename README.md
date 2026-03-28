# Lab Experiments

`lab-experiments` is the incubation repository for exploratory work that is not yet ready for promotion into a primary project. It exists to keep prototypes, speculative implementations, and partial research threads out of the mainline repositories until they either prove themselves and graduate or are intentionally left as archived experiments.

## Scope

- Primary: experimental work with uncertain outcome.
- Includes: `jepa_intuition_poc/`, `geodesic_descent/`, and `personal_ai/`.
- Excludes: stable mainline research, governance runtime infrastructure, and reusable operational manuals.

## Non-goals

- Serving as a permanent storage location for everything that does not fit elsewhere.
- Holding code that already requires mainline maintenance guarantees.
- Competing with `cognitive-lab`, `mtp-weaver`, or `project-manuals` as a stable home.

## Inputs

- New experiments that still need shape, evidence, or architectural direction.
- Partial or archive-worthy work whose value is still exploratory.
- Small prototypes that need room to fail without contaminating the mainline repositories.

## Current Layout

- `jepa_intuition_poc/`: prototype intuition work
- `geodesic_descent/`: terminated experiment kept as a documented record
- `personal_ai/`: exploratory personal research utilities and scripts

## Outputs

- Promotion candidates with a clearer problem statement and a repeatable validation path.
- Archived experiments whose limits are documented rather than silently forgotten.
- A lower-risk place to test ideas before they acquire a long-term maintenance burden.

## Validation

- No single repository-wide validation path is enforced yet.
- Validate each experiment in its own directory before promotion.

## Promotion Path

- Inbound: early or uncertain work that has not yet earned a durable repository boundary.
- Outbound to `cognitive-lab`: experiments that have a stable research question, a repeatable validation path, and evidence that they deserve continued maintenance.
- Outbound to a dedicated repository: experiments that define their own project identity and no longer belong under a shared incubation umbrella.
- Repository role: incubation repository with selective, experiment-by-experiment maintenance.

## Promotion Criteria

- The experiment has a one-paragraph problem statement that survives contact with implementation.
- There is at least one repeatable validation command or script.
- The directory structure is coherent enough that a new reader can tell where the active artifact lives.
- The experiment either aligns with an existing destination repository or justifies a new one.
- If none of the above hold, the experiment stays here or is explicitly archived instead of being promoted.
