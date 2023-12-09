# Projekt zaliczeniowy z przedmiotu "Uczenie Głębokie"

## Opis

Projekt polega na zaprojektowaniu i wykonaniu pracy naukowej z zakresu uczenia głębokiego. Temat pracy obejmuje własną propozycje pracy
badawczej.

## Podział

Praca została podzielona na 4 części i 4 terminy:

1. Propozycja projektu - realizowana do 22.11.2023.
2. Wymagania minimalne - realizowane do 13.12.2023.
3. Wymagania końcowe - realizowane do 24.01.2024.
4. Prezentacja wyników - realizowane do 31.01.2024.

## Propozycja

Temat pracy został zawarty w pliku [Proposal](proposal.pdf).

## Wymagania minimalne

Wstępne wymagania obejmują wykonanie następujących zadań:

- Przygotowanie architektury GAN (Generative Adversarial Network) niezbędne do realizacji zadania. Architektura musi umożliwiać na
  potencjalną transformację przestrzeni ukrytej w dowolną inną.
- Pobranie i przygotowanie danych do uczenia. Zbiór danych musi zawierać wiele etykiet na obraz.
- Przygotowanie skryptu do przygotowywania modelu.
- Przygotowanie TensorBoard'a do interpretacji modelu.

## Skrypty

Skrypty znajdują się w folderze [scripts](scripts).
Do rozruchu skryptów został wykorzystany `pnpm`, zaleca się instalację przez [Node Version Manager](https://github.com/nvm-sh/nvm).

### Opis skryptów

- `pnpm run download:dataset:celeba` - Służy do pobrania zbioru
  danych [CelebA (Large-scale CelebFaces Attributes)](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).
- `pnpm run download:dataset:mnist` - Służy do pobrania zbioru danych [AwA2 (Animals with Attributes 2)](https://cvml.ista.ac.at/AwA2/).
- `pnpm run download:dataset:sun` - Służy do pobrania zbioru
  danych [SUN (Sun Attribute)](https://cs.brown.edu/~gmpatter/sunattributes.html).
- `pnpm run download:datasets` - Służy do pobrania wszystkich zbiorów danych.

### Instalacja pnpm

> npm install -g pnpm

Instalacja zależności

> pnpm install

###### Do wykonywania skryptów zaleca się użycie `WSL2` [Windows Subsystem for Linux](https://learn.microsoft.com/en-us/windows/wsl/install), gdy skrypty są uruchamiane na systemie Windows.
