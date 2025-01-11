# NN data

## collection
- [x] run games with random choices
- [x] at each choice, append all reprs (efficiently?) into dataframe
- [x] record game outcome (win/loss, score, final floor) next to each datapoint in dataframe for game
- [x] concat a bunch of games, dump into parquet
- [x] follow through on randomplayouts.Choice change into set of action types with associated data


## reprs
- [x] deck repr: list of card ids
- [ ] deck repr: and upgrade status
- [ ] map repr: include current pos, burning elite pos
- [ ] relics repr: list of relics including counters if relevant
- [ ] potion repr
- [ ] history repr: list of enemies, events, history of rares and potions being offered
- [ ] choice repr: event type, reward identities, ...

# NN training

- [ ] go from V to Q - outcome prediction for each choice

# battle search
## randomness
rerandomize is cumbersome, need to get an LLM to do it....
instead for now just record and replay randomness

## randomness in search
correlated randomness in parallel nodes

## graph search
implement hashing
