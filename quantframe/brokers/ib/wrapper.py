import threading
from ibapi.wrapper import EWrapper


class IBWrapper(EWrapper):
    """Custom wrapper for Interactive Brokers API."""
    
    def __init__(self):
        EWrapper.__init__(self)
        self.next_valid_order_id = None
        self.historical_data = {}
        self.market_data = {}
        self.streaming_data = {}
        self.stream_event = threading.Event()
        self.account_values = {}
        self.positions = {}
        self.account_pnl = {}

    def error(self, req_id, error_code, error_string, advanced_order_reject_json=""):
        """Handle error messages from IB API."""
        print(f"Error {error_code}: {error_string}")

    def nextValidId(self, order_id):
        """Handle next valid order ID from IB."""
        super().nextValidId(order_id)
        self.next_valid_order_id = order_id

    def updateAccountValue(self, key, val, currency, account):
        """Handle account value updates."""
        try:
            val_ = float(val)
        except:
            val_ = val
        self.account_values[key] = (val_, currency)

    def updatePortfolio(self, contract, position, market_price, market_value,
                       average_cost, unrealized_pnl, realized_pnl, account_name):
        """Handle portfolio updates."""
        portfolio_data = {
            "contract": contract,
            "symbol": contract.symbol,
            "position": position,
            "market_price": market_price,
            "market_value": market_value,
            "average_cost": average_cost,
            "unrealized_pnl": unrealized_pnl,
            "realized_pnl": realized_pnl,
        }
        self.positions[contract.symbol] = portfolio_data

    def pnl(self, req_id, daily_pnl, unrealized_pnl, realized_pnl):
        """Handle PnL updates."""
        pnl_data = {
            "daily_pnl": daily_pnl,
            "unrealized_pnl": unrealized_pnl,
            "realized_pnl": realized_pnl
        }
        self.account_pnl[req_id] = pnl_data

    def orderStatus(self, order_id, status, filled, remaining, avg_fill_price,
                   perm_id, parent_id, last_fill_price, client_id, why_held, mkt_cap_price):
        """Handle order status updates."""
        print(
            f"Order {order_id} Status: {status}, Filled: {filled}, "
            f"Remaining: {remaining}, Last Fill Price: {last_fill_price}"
        )

    def openOrder(self, order_id, contract, order, order_state):
        """Handle open order information."""
        print(
            f"Open Order {order_id}: {contract.symbol} {contract.secType} @ {contract.exchange}: "
            f"{order.action} {order.orderType} {order.totalQuantity} Status: {order_state.status}"
        )
