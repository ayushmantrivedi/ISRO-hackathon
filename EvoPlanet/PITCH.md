# 🎤 EvoPlanet Pitch (ISRO Hackathon)

## 1. The Hook
"Finding exoplanets isn't just about spotting dips in starlight anymore. It's about discerning a planet's shadow from a cosmic ray, a background eclipsing binary, or satellite jitter. Today, astronomers rely on massive algorithmic pipelines and hundreds of hours of manual vetting. What if we could automate this triage with the same rigor as an astrophysicist?"

## 2. The Problem
"Current AI models in astronomy often treat light curves like generic 1D audio waves. They ignore the physics. They ignore the size of the host star. They ignore the movement of the light centroid on the telescope's CCD. By ignoring the context, these basic models trigger thousands of false positives, wasting valuable telescope follow-up time."

## 3. The Solution: EvoPlanet v2
"Enter EvoPlanet v2. We built an advanced **Multi-Modal AI Pipeline** that doesn't just look at the light curve. It looks at the whole picture. 
1. **Multi-Channel Sequence (Tier 1):** Our model simultaneously ingests Flux, Background brightness, NASA Quality flags, and X/Y Centroid motion. This allows the neural network to immediately rule out instrument noise and eclipsing binaries natively, before they are ever flagged.
2. **Global Physics Context (Tier 2 & 3):** We inject an 8-dimensional astrophysical metadata vector directly into the neural network's decision layer. By combining the star's static properties (Radius, Mass, Temperature) with dynamically calculated Box Least Squares (BLS) periodograms, the model interprets the light curve *in context*—knowing that a 1% dip on a giant star means something completely different than a 1% dip on a red dwarf.
3. **EvoPhysicsInspector:** We don't just rely on statistics. We built a dynamic physics consistency layer that learns the boundary conditions of reality. If a neural network predicts a transit duration that is physically impossible for the star's size, the inspector instantly overrides and penalizes the false positive.
4. **Evolutionary NAS (Phase 4):** Instead of guessing the best architecture, our system uses an Evolutionary Neural Architecture Search (EvoNet) to dynamically evolve the optimal way to fuse these physics features together."

## 4. The Impact
"EvoPlanet isn't a black box. It was stress-tested against notoriously ambiguous signals like KIC 12557548. It ranks candidates using Epistemic Uncertainty via Monte Carlo Dropout, and provides visual Saliency Maps to tell astronomers exactly *why* it flagged a transit. We aren't just building a classifier; we are bringing production-grade, interpretable AI to the hunt for Earth 2.0."
