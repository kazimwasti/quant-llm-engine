# Paper-Trading Operational Test Plan

The offline acceptance tests cover the safety boundary before an approved
`ExecutionPayload` reaches a broker. Run them before each deploy:

```powershell
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

They verify the Phase F5 conditions from the supplied workflow:

- a disconnect blocks new orders until an explicit post-reconnect health check;
- client order IDs persist across restarts and concurrent controllers, preventing
  a replay or duplicate;
- a kill switch survives a restart and requires cancellation/flattening;
- the daily-loss cap and consecutive-stopout lock prevent new orders, including
  after a process restart; and
- the controller cannot authorize anything outside paper-trading mode.

## Broker practice-account checklist

Run these only against a practice account after the offline suite passes.

1. Start in `TRADING_MODE=shadow`; confirm signals are logged and no submit path is called.
2. Switch to paper mode, authenticate, reconcile orders and positions, then call
   `mark_connection_healthy()` on the safety controller.
3. During market hours, terminate and restart the service. Reconcile the practice
   account before marking the new process healthy. Confirm the same client order
   ID is rejected locally.
4. Interrupt the REST/WebSocket connection. Call `mark_connection_lost()` and
   confirm no order is authorized until the connection is restored and account
   state has been reconciled.
5. Engage the kill switch. The integration must cancel working orders and flatten
   positions at the broker; retain the lock until that result is confirmed.
6. Exercise daily-loss and stopout limits using the practice account. Record the
   broker timestamps, client order IDs, fills, rejects, latency, and slippage.
   At the next session boundary, reset the risk manager, reconcile account state,
   and explicitly call `clear_session_halt()` before authorizing a new order.

`PaperTradingSafetyController` intentionally does not guess broker endpoint names
for reconnect, cancel, or flatten actions. The broker adapter must perform those
actions and only clear the kill switch after a successful account-state review.
