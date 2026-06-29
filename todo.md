# NN data

## reprs
- [x] deck repr: list of card ids
- [x] deck repr: and upgrade status
- [x] basic relics repr
- [x] potion repr
- [x] basic event choices
- [x] event choice info
- [x] map repr: less crazy version
- [x] reenable path choices
- [x] share embeds between identical IntSpaces (key it) and EnumSpaces. (maybe even FixedVecSpace proj params for gold?)
- [x] map repr: mark reachable nodes from each choice
- [ ] eliminate redundancy when showing card choices from deck and in map choices?
- [ ] shop prices
- [ ] relics repr: add counters
- [ ] history repr: list of enemies, events, history of rares and potions being offered
- [ ] distinguish Prayer Wheel option sets

## actions
- [ ] full potion actions incl discard before gain & drink fruit juice anytime
- [ ] match and keep

# learning

## aux losses
- [ ] PPG
- [ ] winprob
- [ ] HP loss conditioned on enemy: extra seq item that is masked in attention
- [ ] how many E, R, ?, $ are on min and max paths starting from each possible path

## tuning
- [ ] LR decay

## RL
- [ ] exploration: per-run random card preference rewards

# throughput
- [ ] multiprocessing


# battle search
## randomness
- [x] record and replay randomness

## randomness in search
- [x] correlated randomness in parallel nodes

## graph search
- [x] implement hashing

# cleanups
- [ ] refactor comm.py
    - share engine logic instead of reconstructing a bunch of it for the shadow
    - split into multiple components, e.g. shadow state class that's fresh for each battle
- [ ] audit for: correctness, duplicated logic that should be shared, papering over issues rather than root causing, ...

