# NN data

## reprs
- [x] deck repr: list of card ids
- [x] deck repr: and upgrade status
- [x] basic relics repr
- [x] potion repr
- [x] basic event choices
- [x] event choice info
- [ ] map repr: less crazy version, mark choice nodes and all reachable nodes
- [ ] reenable path choices
- [ ] share embeds between identical IntSpaces (key it) and EnumSpaces. (maybe even FixedVecSpace proj params for gold?)
- [ ] eliminate redundancy when showing card choices from deck
- [ ] shop prices
- [ ] relics repr: add counters
- [ ] history repr: list of enemies, events, history of rares and potions being offered

# RL

- [ ] sensitivity.py to analyze behavior and make sure inputs work
- [ ] LR decay
- [ ] exploration: make more-random states and start from them

# optimization

- [ ] multiprocessing


# battle search
## randomness
rerandomize is cumbersome, need to get an LLM to do it....
instead for now just record and replay randomness

## randomness in search
correlated randomness in parallel nodes

## graph search?
implement hashing
