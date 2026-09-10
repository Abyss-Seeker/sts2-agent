# Native UI parity audit (PHASE B2)

Invariant for the formal headful benchmark:

```text
headful_native_ui = true  =>  CardSelectCmd.Selector == null
```

`RlAutoSlayer.PlayRunAsync` only installs `RlCardSelector` when
`HeadfulMode == false`; headful logs
`Headful: CardSelectCmd.Selector left null (native UI driven by
coordinator)`.

## Screen inventory (real build types)

| Screen type | Examples | Coordinator |
| --- | --- | --- |
| `NPlayerHand` selection mode | Headbutt discard, hand upgrade, potion-triggered hand pick | yes (Path 1) |
| `NChooseACardSelectionScreen` | generated card / discover | yes (Path 2) |
| `NSimpleCardSelectScreen` | potion selection, simple pick, upgrade/remove/transform | yes (Path 3, generic grid) |
| `NDeckCardSelectScreen` | deck upgrade / remove / transform / enchant | yes (Path 3, generic grid) |
| `NChooseABundleSelectionScreen` | card bundle (rewards) | RL screen handler (not a card-select grid) |
| `NChooseARelicSelection` | relic choice | RL screen handler |

## Coordinator: `RlNativeSelectionCoordinator`

State machine per screen instance: `Idle -> Observed (screen marked seen)
-> RequestSent (one card_select per instance) -> AwaitingAgent ->
ApplyingClicks -> AwaitingClose`.

* Bounded 200 ms polling (`PollInterval`), single-slot `SemaphoreSlim`
  gate, no busy loop (§19).
* `_overlaySeen` / `_handSeenInSelectMode` guarantee **one** bridge
  request per screen instance (§18).
* Ownership: `OwnsOverlay()` covers choose-a-card / simple / deck screens
  so the outer AutoSlay drain only waits and never chooses (§20).
* Clicks go through `NCardHolder._hitbox.ForceClick` (the same control a
  human clicks) -- never `ICardSelector`, never a manual
  `TaskCompletionSource` (§15).
* Multi-select: the agent replies `{"indexes":[i,j,...]}`; the
  coordinator clicks each native holder and then the REAL confirm button
  (`_confirmButton` or any visible `NButton` whose name mentions
  "confirm"). No backend submit (§22).
* Observation fields are human-visible only: name, rendered text, cost,
  upgrade preview, enchant/affliction, prompt, min/max, combat context
  (§25). No draw order, no RNG, no future values.

## Native auto-select (allowed)

If the game decides `available <= min_required` and
`RequireManualConfirmation == false`, it may resolve without opening a
screen. That is native behaviour, not a bridge bypass (§24) -- nothing to
audit beyond noting it.

## Bypass detection

If headful ever observes `CardSelectCmd.Selector != null` the run must log
`NATIVE_UI_BYPASS` and set `benchmark_valid=false` (§21). The Python side
treats any unexpected selector state as a purity failure.

## VERIFIED THIS ROUND

Headbutt / potion / deck / multi-select: see the PHASE C targeted smoke
results in the final gate report (status per case: PASS / FAIL /
NOT TESTED).
