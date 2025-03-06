from .wrapper import IBWrapper
from .client import IBClient


class IBConnection:
    """Main connection class for Interactive Brokers API."""
    
    def __init__(self, host="127.0.0.1", port=7497, client_id=1, account=None):
        """Initialize IB connection."""
        self.wrapper = IBWrapper()
        self.client = IBClient(self.wrapper)
        self.host = host
        self.port = port
        self.client_id = client_id
        self.account = account
        self._connected = False

    def connect(self):
        """Connect to IB TWS/Gateway."""
        if not self._connected:
            self.client.connect(
                host=self.host,
                port=self.port,
                client_id=self.client_id,
                account=self.account
            )
            self._connected = True

    def disconnect(self):
        """Disconnect from IB TWS/Gateway."""
        if self._connected:
            self.client.disconnect()
            self._connected = False

    def is_connected(self):
        """Check if connected to IB."""
        return self._connected

    def get_account_values(self, key=None):
        """Get account values."""
        return self.client.get_account_values(key)

    def get_positions(self):
        """Get current positions."""
        return self.client.get_positions()

    def get_pnl(self, req_id=1):
        """Get PnL information."""
        return self.client.get_pnl(req_id)

    def send_order(self, symbol, action, quantity, order_type="MKT", 
                  limit_price=None, stop_price=None, sec_type="STK", 
                  exchange="SMART", currency="USD"):
        """Send an order to IB."""
        contract = self.client.create_contract(
            symbol=symbol,
            sec_type=sec_type,
            exchange=exchange,
            currency=currency
        )
        
        order = self.client.create_order(
            action=action,
            quantity=quantity,
            order_type=order_type,
            limit_price=limit_price,
            stop_price=stop_price
        )
        
        return self.client.send_order(contract, order)

    def cancel_all_orders(self):
        """Cancel all open orders."""
        self.client.cancel_all_orders()

    def cancel_order(self, order_id):
        """Cancel a specific order."""
        self.client.cancel_order_by_id(order_id)

    def update_order(self, order_id, symbol, action, quantity, order_type="MKT",
                    limit_price=None, stop_price=None, sec_type="STK",
                    exchange="SMART", currency="USD"):
        """Update an existing order."""
        contract = self.client.create_contract(
            symbol=symbol,
            sec_type=sec_type,
            exchange=exchange,
            currency=currency
        )
        
        order = self.client.create_order(
            action=action,
            quantity=quantity,
            order_type=order_type,
            limit_price=limit_price,
            stop_price=stop_price
        )
        
        return self.client.update_order(contract, order, order_id)
