import time
import threading
from ibapi.client import EClient
from ibapi.contract import Contract
from ibapi.order import Order


class IBClient(EClient):
    """Custom client for Interactive Brokers API."""
    
    def __init__(self, wrapper):
        EClient.__init__(self, wrapper)
        self.account = None
        
    def connect(self, host="127.0.0.1", port=7497, client_id=1, account=None):
        """Connect to IB TWS/Gateway."""
        self.account = account
        super().connect(host, port, client_id)
        thread = threading.Thread(target=self.run, daemon=True)
        thread.start()
        time.sleep(1)  # Give time for connection to establish

    def get_account_values(self, key=None):
        """Get account values from IB."""
        self.reqAccountUpdates(True, self.account)
        time.sleep(2)
        if key:
            return self.wrapper.account_values.get(key)
        return self.wrapper.account_values

    def get_positions(self):
        """Get current positions from IB."""
        self.reqAccountUpdates(True, self.account)
        time.sleep(2)
        return self.wrapper.positions

    def get_pnl(self, req_id=1):
        """Get PnL information from IB."""
        self.reqPnL(req_id, self.account, "")
        time.sleep(2)
        return self.wrapper.account_pnl

    def send_order(self, contract, order):
        """Send an order to IB."""
        order_id = self.wrapper.next_valid_order_id
        self.placeOrder(orderId=order_id, contract=contract, order=order)
        self.reqIds(-1)  # Request next valid ID
        return order_id

    def cancel_all_orders(self):
        """Cancel all open orders."""
        self.reqGlobalCancel()

    def cancel_order_by_id(self, order_id):
        """Cancel a specific order by ID."""
        self.cancelOrder(orderId=order_id, manualCancelOrderTime="")

    def update_order(self, contract, order, order_id):
        """Update an existing order."""
        self.cancel_order_by_id(order_id)
        return self.send_order(contract, order)

    @staticmethod
    def create_contract(symbol, sec_type="STK", exchange="SMART", currency="USD", 
                       expiry=None, strike=None, right=None):
        """Create a contract object."""
        contract = Contract()
        contract.symbol = symbol
        contract.secType = sec_type
        contract.exchange = exchange
        contract.currency = currency
        
        if expiry:
            contract.lastTradeDateOrContractMonth = expiry
        if strike:
            contract.strike = strike
        if right:
            contract.right = right
            
        return contract

    @staticmethod
    def create_order(action, quantity, order_type="MKT", limit_price=None, stop_price=None):
        """Create an order object."""
        order = Order()
        order.action = action
        order.totalQuantity = quantity
        order.orderType = order_type
        
        if limit_price:
            order.lmtPrice = limit_price
        if stop_price:
            order.auxPrice = stop_price
            
        return order
