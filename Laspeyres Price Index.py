import pandas as pd

def calculate_laspeyres_index(base_data, current_data):
    """
    Calculates the Laspeyres Price Index.
    base_data: DataFrame with 'ticker', 'price_base', 'quantity_base'
    current_data: DataFrame with 'ticker', 'price_current'
    """
    # Merge datasets on ticker to ensure we are comparing the same assets
    df = pd.merge(base_data, current_data, on='ticker')
    
    # Calculate Numerator: Current Price * Base Quantity
    numerator = (df['price_current'] * df['quantity_base']).sum()
    
    # Calculate Denominator: Base Price * Base Quantity
    denominator = (df['price_base'] * df['quantity_base']).sum()
    
    # Calculate Index
    laspeyres_index = (numerator / denominator) * 100
    
    return round(laspeyres_index, 2)

# --- Example Usage ---
data_base = {
    'ticker': ['AAPL', 'MSFT', 'GOOGL'],
    'price_base': [150.00, 300.00, 2800.00],
    'quantity_base': [1000, 500, 100] # Fixed basket from base period
}

data_current = {
    'ticker': ['AAPL', 'MSFT', 'GOOGL'],
    'price_current': [155.00, 310.00, 2750.00] # New prices
}

df_base = pd.DataFrame(data_base)
df_current = pd.DataFrame(data_current)

index_value = calculate_laspeyres_index(df_base, df_current)
print(f"The Laspeyres Index Value is: {index_value}")
