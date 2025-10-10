import os
from binance.client import Client

apikey = os.environ['RPUc9J5YZQFaRIBbOMViVB36rdFY5LqfrSVraPVE5fDtbf6E7SY5Dnkbdlie9LhG']
apisecret = os.environ['CJTsNHxoH36700OOzyB3mxeRcFuc5salVfQ0nb0gtPBa2oWli6C24wO2rRjQ6gq3']
client = Client(apikey, apisecret, testnet=True)

# client = Client(api_key, api_secret, tld='us')

