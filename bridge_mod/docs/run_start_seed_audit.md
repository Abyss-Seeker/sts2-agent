# Run start & seed audit (PHASE B1)

Goal: the formal paired A/B must be able to start **the same seed** under
both decision modes. This document records what the real game build
exposes (investigated, not guessed).

## Where a run actually starts

| Concern | Real API / code |
| --- | --- |
| Mod entry | `MainFile.Initialize()` -> `LaunchRlAutoSlayAsync()` -> `new RlAutoSlayer().Start(seed)` |
| Seed source at boot | `SeedHelper.GetRandomSeed()` (random each launch) |
| Run loop | `RlAutoSlayer.RunAsync(seed)` -> `PlayRunAsync(seed)` |
| Seed applied to the game | `NGame.Instance.DebugSeedOverride = seed` (type: `string`) |
| Deterministic RNG | `new Rng((uint)StringHelper.GetDeterministicHashCode(seed))` |
| Main menu flow | `PlayMainMenuAsync()`: resume saved run -> else abandon run -> Singleplayer -> CharacterSelectScreen -> select character -> Confirm |
| Character | `PreferredCharacterId = "Ironclad"` (first unlocked matching character) |
| Difficulty / ascension | standard run (ascension 0); not exposed by the current flow |
| Save lifecycle | `SaveManager` handles run saves; `ResumeSavedRun` may resurrect a save at main menu |
| Abandon / restart | `AbandonRunAsync()` clicks options -> abandon -> proceed |

## Bridge protocol added (B1)

Request (Python -> bridge):

```json
{"action": "start_run", "seed": "ABC123", "character": "Ironclad", "difficulty": 0}
```

Behaviour: `RlAutoSlayer.StartRunAsync(seed, ct)`

1. `Stop()` the current run loop (same STS2 process, same TCP client).
2. best-effort `AbandonRunAsync()` (no-op when already on the main menu).
3. temporarily disable `ResumeSavedRun` so the abandoned save cannot be
   resurrected.
4. `Start(seed)` -> sets `DebugSeedOverride`, walks the main menu into a
   fresh run.
5. waits until `NGame.Instance.DebugSeedOverride == seed` and a `RunState`
   exists.

Acknowledgement (bridge -> Python, sent as a state message):

```json
{"type":"start_run_ack","success":true,"requested_seed":"ABC123",
 "actual_seed":"ABC123","seed_match":true,"character":"Ironclad",
 "difficulty":0,"floor":0,"error":""}
```

`seed_applied_to_game` is only true when `success && seed_match`.
Otherwise the runner invalidates the run with `SEED_MISMATCH`.

## Human parity

The seed is **benchmark controller metadata**. It is never put into the
LLM observation: the Python agent consumes `start_run_ack` before screen
routing and `continue`s, so the model can never use a seed to predict
draw order or RNG.

## Not game-side restart

`start_run` never restarts STS2 and never drops the bridge connection:
paired tasks reuse one game process (see PHASE B/C launch accounting).
