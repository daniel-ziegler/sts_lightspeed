# NN data collection

## collection
* run games with random choices
* at each choice, append all reprs (efficiently?) into dataframe
* record game outcome (win/loss, score, final floor) next to each datapoint in dataframe for game
* concat a bunch of games, dump into parquet
- follow through on randomplayouts.Choice change into set of action types with associated data


## reprs
- map repr: include current pos, burning elite pos
- deck repr: list of card ids and upgrade status
- relics repr: list of relics including counters if relevant
- potion repr
- history repr: list of enemies, events, history of rares and potions being offered
- choice repr: event type, reward identities, ...

# battle search
## randomness
rerandomize is cumbersome, need to get an LLM to do it....
instead for now just record and replay randomness

## randomness in search
correlated randomness in parallel nodes

## graph search
implement hashing
