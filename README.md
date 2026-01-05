# Cognitive Limits and the Emergence of Logic — Toy Model

Repository associated with the article [**What You Can't Ask: Cognitive Limits and the Emergence of Logic**](<insert-link-here>), which explores how internal architecture and sampling shape the logical capabilities of a mind. I use simple toy models to simulate how different "minds" (learners) can or cannot recover the logical structure of a world made of bit strings.

## Overview

I define a simple **Ehrenfeucht–Fraïssé (EF) game** implementation to define indistinguishability under different logics.

The repo contains two main types of learners:

- **Classifier Learner**: Trained on labeled examples (e.g., whether a string has even parity), it learns to predict logical properties directly.
- **Contrastive Learner**: Trained on *positive* and *negative* pairs of strings, it learns an embedding space where similar strings are close together.

## Key Concepts

- **Toy Universe**: Strings of 4 bits (e.g., `1101`, `0010`) and their underlying logical structure.
- **EF Game**: Models which strings are indistinguishable under limited logic (e.g., first-order logic without counting).
- **Contrastive Learning**: Learning without labels, using only local associations.
- **Cognitive Limits**: The idea that what a mind can learn depends on both its architecture and its access to the world.

## Repo Structure

├── notebooks/
│ ├── Ehrenfeucht–Fraisse_game.ipynb # Notebook demonstrating the EF game logic used for this experiment
│ ├── classifier_contrastive_learners.ipynb # Notebook for contrastive and classifier learning exploration
├── src/
│ ├── ef_game.py    # EF game logic and structure definitions
│ ├── models.py # Classifier and contrastive learner models
│ ├── utils.py # Some functions for plotting and graph generationm
├── README.md
