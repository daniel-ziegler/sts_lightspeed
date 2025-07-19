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
- [ ] map repr: mark reachable nodes from each choice
- [ ] eliminate redundancy when showing card choices from deck and in map choices
- [ ] shop prices
- [ ] relics repr: add counters
- [ ] history repr: list of enemies, events, history of rares and potions being offered

# learning

## aux losses
- [ ] PPG
- [ ] winprob
- [ ] HP loss conditioned on enemy: extra seq item that is masked in attention
- [ ] how many E, R, ?, $ are on min and max paths starting from each possible path

## tuning
- [ ] LR decay

## RL
- [ ] exploration: make more-random states and start from them

## debugging
- [ ] sensitivity.py to analyze behavior and make sure inputs work

# throughput
- [ ] multiprocessing


# battle search
## randomness
rerandomize is cumbersome, need to get an LLM to do it....
instead for now just record and replay randomness

## randomness in search
correlated randomness in parallel nodes

## graph search?
implement hashing
